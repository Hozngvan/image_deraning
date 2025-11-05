import torch
from torch.utils.data import DataLoader
import random
import threading
import queue

def _ms_loop(dataset, index_queue, data_queue, collate_fn, scale, seed, worker_id):
    """
    Worker loop: load data and apply multi-scale if needed.
    """
    torch.set_num_threads(1)
    torch.manual_seed(seed)

    while True:
        task = index_queue.get()
        if task is None:
            break
        idx, batch_indices = task
        try:
            idx_scale = 0
            if len(scale) > 1 and getattr(dataset, "train", False):
                idx_scale = random.randrange(0, len(scale))
                dataset.set_scale(idx_scale)

            samples = collate_fn([dataset[i] for i in batch_indices])
            samples.append(idx_scale)
        except Exception as e:
            data_queue.put((idx, e))
        else:
            data_queue.put((idx, samples))


class _MSDataLoaderIter:
    """
    A simplified custom DataLoader iterator compatible with new PyTorch.
    """
    def __init__(self, loader):
        self.loader = loader
        self.dataset = loader.dataset
        self.scale = loader.scale
        self.collate_fn = loader.collate_fn
        self.batch_sampler = loader.batch_sampler
        self.num_workers = loader.num_workers
        self.pin_memory = loader.pin_memory and torch.cuda.is_available()

        self.sample_iter = iter(self.batch_sampler)
        self.data_queue = queue.Queue()
        self.index_queue = queue.Queue()
        self.done_event = threading.Event()
        self.threads = []
        self.seed = torch.randint(0, 2**31, (1,)).item()

        # spawn worker threads
        for worker_id in range(self.num_workers):
            t = threading.Thread(
                target=_ms_loop,
                args=(self.dataset, self.index_queue, self.data_queue,
                      self.collate_fn, self.scale, self.seed + worker_id, worker_id)
            )
            t.daemon = True
            t.start()
            self.threads.append(t)

        # preload indices
        self._prefetch()

    def _prefetch(self):
        """Preload indices into queue"""
        for _ in range(2 * self.num_workers):
            try:
                idx = next(self.sample_iter)
                self.index_queue.put((_, idx))
            except StopIteration:
                break

    def __iter__(self):
        return self

    def __next__(self):
        try:
            idx, data = self.data_queue.get(timeout=30)
        except queue.Empty:
            self._shutdown_workers()
            raise StopIteration

        if isinstance(data, Exception):
            raise data
        return data

    def _shutdown_workers(self):
        if not self.done_event.is_set():
            for _ in range(self.num_workers):
                self.index_queue.put(None)
            self.done_event.set()


class MSDataLoader(DataLoader):
    """
    Multi-scale DataLoader wrapper, compatible with PyTorch 2.x.
    """
    def __init__(
        self, args, dataset, batch_size=1, shuffle=False,
        sampler=None, batch_sampler=None,
        collate_fn=None, pin_memory=False, drop_last=False,
        timeout=0, worker_init_fn=None
    ):
        super().__init__(
            dataset, batch_size=batch_size, shuffle=shuffle,
            sampler=sampler, batch_sampler=batch_sampler,
            num_workers=args.n_threads, collate_fn=collate_fn,
            pin_memory=pin_memory, drop_last=drop_last,
            timeout=timeout, worker_init_fn=worker_init_fn
        )
        self.scale = args.scale

    def __iter__(self):
        return _MSDataLoaderIter(self)
