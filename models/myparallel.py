from torch.nn.parallel import DataParallel
import torch
from torch.nn.parallel._functions import Scatter
from torch.nn.parallel.parallel_apply import parallel_apply

def scatter(inputs, target_gpus, chunk_sizes, dim=0):
    r"""
    Slices tensors into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not tensors.
    """
    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            try:
                return Scatter.apply(target_gpus, chunk_sizes, dim, obj)
            except:
                print('obj', obj.size())
                print('dim', dim)
                print('chunk_sizes', chunk_sizes)
                quit()
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for targets in target_gpus]
    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None

def scatter_kwargs(inputs, kwargs, target_gpus, chunk_sizes, dim=0):
    r"""Scatter with support for kwargs dictionary"""
    if 'gts' in kwargs.keys():
        vertices, faces = kwargs['gts'][-2:]
        kwargs['gts'] = kwargs['gts'][:-2]
        assert len(vertices) == len(target_gpus)
    inputs = scatter(inputs, target_gpus, chunk_sizes, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, chunk_sizes, dim) if kwargs else []
    if 'gts' in kwargs[0].keys():
        for i in range(len(target_gpus)):
            kwargs[i]['gts'].append(vertices[i].to(target_gpus[i]))
            kwargs[i]['gts'].append(faces[i].to(target_gpus[i]))

    assert len(inputs) == len(kwargs)

    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


class MyParallel(DataParallel):
    def __init__(self, *args, **kwargs):
        super(MyParallel, self).__init__(*args, **kwargs)

    def forward(self, *inputs, **kwargs):
        if not self.device_ids or len(self.device_ids) == 1 or inputs[0].shape[self.dim] == 1:
            return self.module(*inputs, **kwargs)

        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        replicas = self.replicate(self.module, self.device_ids)
        outputs = self.parallel_apply(replicas, self.device_ids, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def parallel_apply(self, replicas, device_ids, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, device_ids)

    def scatter(self, inputs, kwargs, device_ids):
        bsz = inputs[0].shape[self.dim]
        chunk_sizes = [bsz // len(self.device_ids)] * len(self.device_ids)
        return scatter_kwargs(inputs, kwargs, device_ids, chunk_sizes, dim=self.dim)