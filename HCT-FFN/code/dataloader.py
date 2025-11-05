import sys
import threading
import queue
import random
import collections

import torch
import torch.multiprocessing as multiprocessing
from torch.utils.data.dataloader import DataLoader, default_collate

class _MSDataLoaderIter:
    def __init__(self, loader):
        self.dataset = loader.dataset
        self.scale = loader.scale
        self.collate_fn = loader.collate_fn
        self.batch_sampler = loader.batch_sampler
        self.num_workers = loader.num_workers
        self.pin_memory = loader.pin_memory and torch.cuda.is_available()
        self.timeout = loader.timeout

        self.sample_iter = iter(self.batch_sampler)
        self.done_event = threading.Event()
        self.data_queue = queue.Queue()
        self.index_queue = queue.Queue()

        # tạo các thread worker (thay vì process)
        for i in range(self.num_workers):
            t = threading.Thread(
                target=_ms_loop,
                args=(self.dataset, self.index_queue, self.data_queue,
                      self.collate_fn, self.scale, torch.randint(0, 2**31, (1,)).item(), None, i)
            )
            t.daemon = True
            t.start()

        # nạp trước index
        self._prefetch()

    def _prefetch(self):
        for i, batch in enumerate(self.batch_sampler):
            self.index_queue.put((i, batch))

    def __iter__(self):
        return self

    def __next__(self):
        if self.data_queue.empty():
            raise StopIteration
        idx, batch = self.data_queue.get()
        if isinstance(batch, Exception):
            raise batch
        return batch

class MSDataLoader(DataLoader):
    def __init__(self, args, dataset, **kwargs):
        super().__init__(dataset, num_workers=args.n_threads, **kwargs)
        self.scale = args.scale

    def __iter__(self):
        return _MSDataLoaderIter(self)
