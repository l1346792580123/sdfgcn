import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from .deephuman import DeepHuman
from .renderpeople import RenderPeople
import torch

# variable vertices length need custom collect_fn
def custom_collate_fn(batch):
    '''
    :param batch B N
    '''
    batch_size = len(batch)
    length = len(batch[0])
    ret = []
    for i in range(length-2):
        data = [torch.from_numpy(item[i]) for item in batch]
        ret.append(torch.stack(data,dim=0))

    vertices = [torch.from_numpy(item[-2]).unsqueeze(0) for item in batch]
    faces = [torch.from_numpy(item[-1]).unsqueeze(0) for item in batch]
    ret.append(vertices)
    ret.append(faces)
    return ret
    

class TotalDataset(object):
    def __init__(self, config, num_iter):
        self.datasets = []
        self.loaders = []
        self.iters = []
        self.num_iter = num_iter
        for key in config.keys():
            dataset = eval(key)(**config[key])
            print(len(dataset))
            # loader = DataLoader(dataset, **config[key]['loader'], collate_fn=custom_collate_fn)
            loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=8, drop_last=True, pin_memory=True, collate_fn=custom_collate_fn)

            self.datasets.append(dataset)
            self.loaders.append(loader)
            self.iters.append(iter(loader))

        self.idx = 0

    def __len__(self):
        return self.num_iter

    def __iter__(self):
        return self

    def change_sigma(self, ratio):
        for dataset in self.datasets:
            dataset.sigma = max(dataset.sigma / ratio, 0.005)

    def __next__(self):
        if self.num_iter <= self.idx:
            raise StopIteration

        datas = []
        for i in range(len(self.iters)):
            try:
                data = next(self.iters[i])
            except:
                self.iters[i] = iter(self.loaders[i])
                data = next(self.iters[i])

            datas.append(data)

        self.idx +=1

        return datas