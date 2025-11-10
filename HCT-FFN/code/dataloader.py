import sys
import threading
import queue
import random

import torch
import torch.multiprocessing as multiprocessing
from torch.utils.data import DataLoader
from torch.utils.data._utils import signal_handling, MP_STATUS_CHECK_INTERVAL
from torch.utils.data._utils.pin_memory import _pin_memory_loop
from torch.utils.data._utils.worker import ExceptionWrapper
from torch.utils.data import default_collate

if sys.version_info[0] == 2:
    import Queue as queue
else:
    import queue


def _ms_loop(dataset, index_queue, data_queue, collate_fn, scale, seed, init_fn, worker_id):
    """Custom worker loop để hỗ trợ multi-scale training."""
    torch.set_num_threads(1)
    random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)
    signal_handling._set_worker_signal_handlers()

    if init_fn is not None:
        init_fn(worker_id)

    while True:
        try:
            r = index_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
        except queue.Empty:
            continue

        if r is None:
            break

        idx, batch_indices = r
        try:
            idx_scale = 0
            if len(scale) > 1 and getattr(dataset, "train", False):
                idx_scale = random.randrange(0, len(scale))
                dataset.set_scale(idx_scale)

            samples = [dataset[i] for i in batch_indices]
            batch = collate_fn(samples)
            batch = batch + (idx_scale,) if isinstance(batch, tuple) else batch + [idx_scale]
            # Đảm bảo luôn là tuple để đồng nhất
            if not isinstance(batch, tuple):
                batch = tuple(batch)    
        except Exception:
            data_queue.put((idx, ExceptionWrapper(sys.exc_info())))
        else:
            data_queue.put((idx, batch))


class _MSDataLoaderIter:
    """Phiên bản tương thích PyTorch 2.x+"""

    def __init__(self, loader):
        self.dataset = loader.dataset
        self.scale = loader.scale
        self.collate_fn = loader.collate_fn
        self.batch_sampler = loader.batch_sampler
        self.num_workers = loader.num_workers
        self.pin_memory = loader.pin_memory and torch.cuda.is_available()
        self.timeout = loader.timeout
        self.worker_init_fn = loader.worker_init_fn

        self.sample_iter = iter(self.batch_sampler)

        self.send_idx = 0
        self.rcvd_idx = 0
        self.reorder_dict = {}
        self.batches_outstanding = 0
        self.shutdown = False
        self.done_event = threading.Event()

        if self.num_workers == 0:
            return

        self.index_queues = [multiprocessing.Queue() for _ in range(self.num_workers)]
        self.worker_result_queue = multiprocessing.SimpleQueue()

        # Seed riêng cho mỗi worker
        base_seed = torch.randint(0, 2**31 - 1, (1,)).item()

        self.workers = [
            multiprocessing.Process(
                target=_ms_loop,
                args=(
                    self.dataset,
                    self.index_queues[i],
                    self.worker_result_queue,
                    self.collate_fn,
                    self.scale,
                    base_seed + i,        # ĐÃ SỬA: + i
                    self.worker_init_fn,
                    i,
                ),
            )
            for i in range(self.num_workers)
        ]

        for w in self.workers:
            w.daemon = True
            w.start()

        # Pin memory thread
        if self.pin_memory:
            self.data_queue = queue.Queue()
            device_id = torch.cuda.current_device()
            self.worker_manager_thread = threading.Thread(
                target=_pin_memory_loop,
                args=(self.worker_result_queue, self.data_queue, self.done_event, device_id),
            )
            self.worker_manager_thread.daemon = True
            self.worker_manager_thread.start()
        else:
            self.data_queue = self.worker_result_queue

        # Prefetch
        for _ in range(2 * self.num_workers):
            self._put_indices()

    def _put_indices(self):
        if self.shutdown:
            return
        try:
            batch_indices = next(self.sample_iter)
        except StopIteration:
            return

        worker_queue = self.index_queues[self.send_idx % self.num_workers]
        worker_queue.put((self.send_idx, batch_indices))
        self.batches_outstanding += 1
        self.send_idx += 1

    def __iter__(self):
        return self

    def __next__(self):
        if self.shutdown:
            raise StopIteration

        # Single-thread mode
        if self.num_workers == 0:
            try:
                batch_indices = next(self.sample_iter)
            except StopIteration:
                raise StopIteration

            # batch = self.collate_fn([self.dataset[i] for i in batch_indices])
            # batch = batch + (0,) if isinstance(batch, tuple) else batch + [0]
            # return tuple(batch) if not isinstance(batch, tuple) else batch
        
            batch = self.collate_fn([self.dataset[i] for i in batch_indices])
           
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                batch = (*batch, 0)
            return batch

        # Multi-worker mode
        while True:
            # Kiểm tra reorder buffer
            if self.rcvd_idx in self.reorder_dict:
                batch = self.reorder_dict.pop(self.rcvd_idx)
                self.rcvd_idx += 1
                self._put_indices()
                self.batches_outstanding -= 1
                return batch

            # Lấy dữ liệu
            try:
                idx, data = self.data_queue.get(timeout=self.timeout if self.timeout > 0 else MP_STATUS_CHECK_INTERVAL)
            except queue.Empty:
                continue
            except Exception:
                # Timeout
                if self.timeout > 0:
                    raise
                continue

            # Xử lý lỗi từ worker
            if isinstance(data, ExceptionWrapper):
                data.reraise()

            # Reorder
            if idx != self.rcvd_idx:
                self.reorder_dict[idx] = data
                continue

            # Đúng thứ tự → trả về batch
            self.rcvd_idx += 1
            self._put_indices()
            self.batches_outstanding -= 1
            return data

    def __del__(self):
        if self.num_workers == 0 or self.shutdown:
            return
        self.shutdown = True
        self.done_event.set()

        for q in self.index_queues:
            try:
                q.put_nowait(None)
            except:
                pass
            q.close()

        for w in self.workers:
            if w.is_alive():
                w.join(timeout=3.0)
                if w.is_alive():
                    w.terminate()


class MSDataLoader(DataLoader):
    """Giữ nguyên API như cũ, đã sửa toàn bộ lỗi."""

    def __init__(
        self,
        args,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        collate_fn=default_collate,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):
        super(MSDataLoader, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=args.n_threads,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )
        self.scale = args.scale

    def __iter__(self):
        return _MSDataLoaderIter(self)