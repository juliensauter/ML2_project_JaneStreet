from torch.utils.data import IterableDataset, get_worker_info
import itertools
import math

from uni2ts.transform import Transformation


class TransformedDataset(IterableDataset):
    
    def __init__(self, base_ds, transform):
        super().__init__()
        self.base_ds = base_ds
        self.transform = transform
    
    def __iter__(self):
        worker_info = get_worker_info()
        iterable = iter(self.base_ds)

        if worker_info is not None:
            print("Multi-process loading. Slicing the iterable...")
            iterable = itertools.islice(
                iterable, worker_info.id, None, worker_info.num_workers
            )
        
        for entry in iterable:
            yield self.transform(entry)

    def __len__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        # Each worker gets a portion of the dataset
        return math.ceil(len(self.base_ds) / num_workers)

class AddFreq(Transformation):
    def __init__(self, freq):
        self.freq = freq
    
    def __call__(self, data):
        data['freq'] = self.freq
        return data