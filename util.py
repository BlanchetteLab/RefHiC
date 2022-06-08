import numpy as np
import numba as nb
import torch

@nb.njit(parallel=True)
def rankdata_core(data):
    """
    parallelized version of scipy.stats.rankdata along  axis 1 in a 2D-array
    """
    ranked = np.empty(data.shape, dtype=np.float64)
    for j in nb.prange(data.shape[0]):
        arr = np.ravel(data[j, :])
        sorter = np.argsort(arr)

        arr = arr[sorter]
        obs = np.concatenate((np.array([True]), arr[1:] != arr[:-1]))

        dense = np.empty(obs.size, dtype=np.int64)
        dense[sorter] = obs.cumsum()

        # cumulative counts of each unique value
        count = np.concatenate((np.nonzero(obs)[0], np.array([len(obs)])))
        ranked[j, :] = count[dense - 1]
    return ranked


def rankdata(data, axis=1):
    """
    parallelized version of scipy.stats.rankdata
    """
    shape = data.shape
    dims = len(shape)
    if axis + 1 > dims:
        raise ValueError('axis does not exist')
    if axis < dims - 1:
        data = np.swapaxes(data, axis, -1)
        shape = data.shape
    if dims > 2:
        data = data.reshape(np.prod(shape[:-1]), shape[-1])

    ranked = rankdata_core(data)

    if dims > 2:
        data = data.reshape(shape)
        ranked = ranked.reshape(shape)
    if axis < dims - 1:
        data = np.swapaxes(data, -1, axis)
        ranked = np.swapaxes(ranked, -1, axis)
    return ranked

def rangeSplit(start,end,size,overlap):
    """
    split a range [start,end) into fixed-size segments with overlaps
    :param start:
    :param end:
    :param size: fixed-size
    :param overlap: overlaps
    :return: list of [start,end)
    """
    start = [i for i in range(start,end,size-overlap)]
    end = [i+size for i in start]
    return list(zip(start[:-1],end[:-1]))


@torch.no_grad()
def update_bn(loader, model, device=None):
    r"""Updates BatchNorm running_mean, running_var buffers in the model.
    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.

    THIS FUNCTION ALLOWS MODEL TAKEN 2 INPUTS

    Args:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.
        device (torch.device, optional): If set, data will be transferred to
            :attr:`device` before being passed into :attr:`model`.
    Example:
        >>> loader, model = ...
        >>> torch.optim.swa_utils.update_bn(loader, model)
    .. note::
        The `update_bn` utility assumes that each data batch in :attr:`loader`
        is either a tensor or a list or tuple of tensors; in the latter case it
        is assumed that :meth:`model.forward()` should be called on the first
        element of the list or tuple corresponding to the data batch.
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    for epoch in range(2):
        for batch_idx,X in enumerate(loader):
            X[1] = X[1].to(device)
            X[2] = X[2].to(device)
            model(X[1],X[2])

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)