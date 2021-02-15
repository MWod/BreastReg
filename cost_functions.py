import math
import numpy as np
import torch as tc
import torch.nn as nn
import torch.nn.functional as F

import utils
import utils_tc


def ncc_global_np(source: np.ndarray, target: np.ndarray, **params):
    if source.shape != target.shape:
        raise ValueError("Resolution of both the images must be the same.")
    source_mean, target_mean = np.mean(source), np.mean(target)
    source_std, target_std = np.std(source), np.std(target)
    ncc = np.mean((source - source_mean) * (target - target_mean) / (source_std * target_std))
    if ncc != ncc:
        ncc = -1
    return -ncc

def ncc_global_tc(sources: tc.Tensor, targets: tc.Tensor, device: str="cpu", **params):
    sources = (sources - tc.min(sources)) / (tc.max(sources) - tc.min(sources))
    targets = (targets - tc.min(targets)) / (tc.max(targets) - tc.min(targets))
    if sources.size() != targets.size():
        raise ValueError("Shape of both the tensors must be the same.")
    size = sources.size()
    prod_size = tc.prod(tc.Tensor(list(size[1:])))
    sources_mean = tc.mean(sources, dim=list(range(1, len(size)))).view((sources.size(0),) + (len(size)-1)*(1,))
    targets_mean = tc.mean(targets, dim=list(range(1, len(size)))).view((targets.size(0),) + (len(size)-1)*(1,))
    sources_std = tc.std(sources, dim=list(range(1, len(size))), unbiased=False).view((sources.size(0),) + (len(size)-1)*(1,))
    targets_std = tc.std(targets, dim=list(range(1, len(size))), unbiased=False).view((targets.size(0),) + (len(size)-1)*(1,))
    ncc = (1 / prod_size) * tc.sum((sources - sources_mean) * (targets - targets_mean) / (sources_std * targets_std), dim=list(range(1, len(size))))
    ncc = tc.mean(ncc)
    if ncc != ncc:
        ncc = tc.autograd.Variable(tc.Tensor([-1]), requires_grad=True).to(device)
    return -ncc

def ncc_local_tc(sources: tc.Tensor, targets: tc.Tensor, device: str="cpu", **params):
    """
    Implementation inspired by VoxelMorph (with some modifications).
    """
    sources = (sources - tc.min(sources)) / (tc.max(sources) - tc.min(sources))
    targets = (targets - tc.min(targets)) / (tc.max(targets) - tc.min(targets))
    ndim = len(sources.size()) - 2
    if ndim not in [2, 3]:
        raise ValueError("Unsupported number of dimensions.")
    try:
        win_size = params['win_size']
    except:
        win_size = 9
    window = (win_size, ) * ndim
    sum_filt = tc.ones([1, 1, *window]).to(device)
    pad_no = math.floor(window[0] / 2)
    stride = ndim * (1,)
    padding = ndim * (pad_no,)
    conv_fn = getattr(F, 'conv%dd' % ndim)
    sources_denom = sources**2
    targets_denom = targets**2
    numerator = sources*targets
    sources_sum = conv_fn(sources, sum_filt, stride=stride, padding=padding)
    targets_sum = conv_fn(targets, sum_filt, stride=stride, padding=padding)
    sources_denom_sum = conv_fn(sources_denom, sum_filt, stride=stride, padding=padding)
    targets_denom_sum = conv_fn(targets_denom, sum_filt, stride=stride, padding=padding)
    numerator_sum = conv_fn(numerator, sum_filt, stride=stride, padding=padding)
    size = np.prod(window)
    u_sources = sources_sum / size
    u_targets = targets_sum / size
    cross = numerator_sum - u_targets * sources_sum - u_sources * targets_sum + u_sources * u_targets * size
    sources_var = sources_denom_sum - 2 * u_sources * sources_sum + u_sources * u_sources * size
    targets_var = targets_denom_sum - 2 * u_targets * targets_sum + u_targets * u_targets * size
    ncc = cross * cross / (sources_var * targets_var + 1e-5)
    return -tc.mean(ncc)

def mse_np(source: np.ndarray, target: np.ndarray, **params):
    if source.shape != target.shape:
        raise ValueError("Resolution of both the images must be the same.")
    return np.mean((source - target)**2)

def mse_tc(sources: tc.Tensor, targets: tc.Tensor, device: str="cpu", **params):
    if sources.size() != targets.size():
        raise ValueError("Shape of both the tensors must be the same.")
    return tc.mean((sources - targets)**2)

def ngf_np(source: np.ndarray, target: np.ndarray, **params):
    epsilon = params['epsilon']
    try:
        return_response = params['return_response']
    except:
        return_response = False
    
    ndim = len*(source.shape)
    if ndim == 2:
        sgx, sgy = np.gradient(source)
        tgx, tgy = np.gradient(target)
        ds = np.sqrt(sgx**2 + sgy**2 + epsilon**2)
        dt = np.sqrt(tgx**2 + tgy**2 + epsilon**2)
        nm = sgx*tgx + sgy*tgy
    elif ndim == 3:
        sgx, sgy, sgz = np.gradient(source)
        tgx, tgy, tgz = np.gradient(target)
        ds = np.sqrt(sgx**2 + sgy**2 + sgz**2 + epsilon**2)
        dt = np.sqrt(tgx**2 + tgy**2 + tgz**2 + epsilon**2)
        nm = sgx*tgx + sgy*tgy + sgz*tgz
    else:
        raise ValueError("Unsupported number of dimensions.")
    response = 1 - (nm / (ds * dt))**2
    ngf = np.mean(response)
    if return_response:
        return ngf, response

def ngf_tc(sources: tc.Tensor, targets: tc.Tensor, device: str="cpu", **params):
    epsilon = params['epsilon']
    try:
        return_response = params['return_response']
    except:
        return_response = False

    ndim = len(sources.size()) - 2
    if ndim == 2:
        sgx, sgy = utils_tc.tensor_gradient(sources, device=device)
        tgx, tgy = utils_tc.tensor_gradient(targets, device=device)
        ds = tc.sqrt(sgx**2 + sgy**2 + epsilon**2)
        dt = tc.sqrt(tgx**2 + tgy**2 + epsilon**2)
        nm = sgx*tgx + sgy*tgy
    elif ndim == 3:
        sgx, sgy, sgz = utils_tc.tensor_gradient(sources, device=device)
        tgx, tgy, tgz = utils_tc.tensor_gradient(targets, device=device)
        ds = tc.sqrt(sgx**2 + sgy**2 + sgz**2 + epsilon**2)
        dt = tc.sqrt(tgx**2 + tgy**2 + tgz**2 + epsilon**2)
        nm = sgx*tgx + sgy*tgy + sgz*tgz
    else:
        raise ValueError("Unsupported number of dimensions.")
    response = 1 - (nm / (ds * dt))**2
    ngf = tc.mean(response)
    if return_response:
        return ngf, response
    else:
        return ngf

def mind_ssc_tc(sources: tc.Tensor, targets: tc.Tensor, device: str="cpu", **params):
    """
    Implementation inspired by https://github.com/voxelmorph/voxelmorph/pull/145 (with some modifications).
    """
    sources = (sources - tc.min(sources)) / (tc.max(sources) - tc.min(sources))
    targets = (targets - tc.min(targets)) / (tc.max(targets) - tc.min(targets))
    try:
        radius = params['radius']
        dilation = params['dilation']
    except:
        radius = 2
        dilation = 2

    ndim = len(sources.size()) - 2
    if ndim not in [2, 3]:
        raise ValueError("Unsupported number of dimensions.")
    if ndim == 2:
        sources = sources.unsqueeze(3)
        targets = targets.unsqueeze(3)

    def pdist_squared(x):
        xx = (x**2).sum(dim=1).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * tc.bmm(x.permute(0, 2, 1), x)
        dist[dist != dist] = 0
        dist = tc.clamp(dist, 0.0, np.inf)
        return dist

    def mind_ssc(images, radius, dilation):
        kernel_size = radius * 2 + 1
        six_neighbourhood = tc.Tensor([[0,1,1],
                                        [1,1,0],
                                        [1,0,1],
                                        [1,1,2],
                                        [2,1,1],
                                        [1,2,1]]).long()
        dist = pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)
        x, y = tc.meshgrid(tc.arange(6), tc.arange(6))
        mask = ((x > y).view(-1) & (dist == 2).view(-1))
        idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1,6,1).view(-1,3)[mask,:]
        idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6,1,1).view(-1,3)[mask,:]
        mshift1 = tc.zeros(12, 1, 3, 3, 3).cuda()
        mshift1.view(-1)[tc.arange(12) * 27 + idx_shift1[:,0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
        mshift2 = tc.zeros(12, 1, 3, 3, 3).cuda()
        mshift2.view(-1)[tc.arange(12) * 27 + idx_shift2[:,0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
        rpad1 = nn.ReplicationPad3d(dilation)
        rpad2 = nn.ReplicationPad3d(radius)
        ssd = F.avg_pool3d(rpad2((F.conv3d(rpad1(images), mshift1, dilation=dilation) - F.conv3d(rpad1(images), mshift2, dilation=dilation)) ** 2), kernel_size, stride=1)
        mind = ssd - tc.min(ssd, 1, keepdim=True)[0]
        mind_var = tc.mean(mind, 1, keepdim=True)
        mind_var = tc.clamp(mind_var, mind_var.mean().item()*0.001, mind_var.mean().item()*1000)
        mind /= mind_var
        mind = tc.exp(-mind)
        mind = mind[:, tc.Tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]
        return mind

    return tc.mean((mind_ssc(sources, radius, dilation) - mind_ssc(targets, radius, dilation))**2)