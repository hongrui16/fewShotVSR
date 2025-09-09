import os
import sys

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dataloader.dataset.dummyDataset import DummyDataset
from dataloader.dataset.REDS import REDS25WindowDataset
from dataloader.dataset.Vimeo90K import Vimeo7to14Dataset



def build_dataloader(
        split = 'train', 
        batch_size= 2, 
        **kwargs,
        ):
    
    num_workers = kwargs.get('num_workers', 4)
    logger = kwargs.get('logger', None)
    device = kwargs.get('device', 'cpu')

    dataset_name = kwargs.get('dataset_name', None)
    if dataset_name == 'DummyDataset':
        dataset = DummyDataset()
    elif dataset_name == 'REDS':
        pass
    elif dataset_name == 'Vimeo-90K':
        dataset = Vimeo7to14Dataset(split=split, logger = logger)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    if split == 'train':
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return dataloader, dataset


if __name__ == "__main__":
    pass