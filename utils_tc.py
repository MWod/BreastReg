import numpy as np
import torch as tc
import torch.nn.functional as F


def scale_penalty(transform):
    sx = tc.sqrt(transform[0, 0, 0]**2 + transform[0, 0, 1]**2 + transform[0, 0, 2]**2)
    sy = tc.sqrt(transform[0, 1, 0]**2 + transform[0, 1, 1]**2 + transform[0, 1, 2]**2)
    sz = tc.sqrt(transform[0, 2, 0]**2 + transform[0, 2, 1]**2 + transform[0, 2, 2]**2)
    penalty = tc.sqrt((sx - 1)**2 + (sy - 1)**2 + (sz - 1)**2)
    return penalty

def resample_to_reg(tensor: tc.Tensor, old_spacing: tuple, new_spacing: tuple, device: str="cpu", mode: str='bilinear'):
    old_size = tensor.size()
    ndim = len(old_size) - 2
    if ndim == 2:
        new_size = (old_size[0], old_size[1], int(old_size[2]* old_spacing[1] / new_spacing[1]), int(old_size[3]* old_spacing[0] / new_spacing[0]))
    elif ndim == 3:
        new_size = (old_size[0], old_size[1], int(old_size[2]* old_spacing[1] / new_spacing[1]), int(old_size[3]* old_spacing[0] / new_spacing[0]), int(old_size[4]* old_spacing[2] / new_spacing[2]))
    resampled_tensor = resample_tensor(tensor, new_size, device=device, mode=mode)
    return resampled_tensor

def warp_tensor(tensor: tc.Tensor, displacement_field: tc.Tensor, grid: tc.Tensor=None, device: str="cpu", mode: str='bilinear'):
    if grid is None:
        grid = generate_grid(tensor.size(), device=device)
    sampling_grid = grid + displacement_field
    transformed_tensor = F.grid_sample(tensor, sampling_grid, mode=mode, padding_mode='zeros', align_corners=False)
    return transformed_tensor

def transform_tensor(tensor: tc.Tensor, sampling_grid: tc.Tensor, grid: tc.Tensor=None, device: str="cpu", mode: str='bilinear'):
    transformed_tensor = F.grid_sample(tensor, sampling_grid, mode=mode, padding_mode='zeros', align_corners=False)
    return transformed_tensor

def resample_tensor(tensor: tc.Tensor, new_size: tc.Tensor, device: str="cpu", mode: str='bilinear'):
    sampling_grid = generate_grid(new_size, device=device)
    resampled_tensor = F.grid_sample(tensor, sampling_grid, mode=mode, padding_mode='zeros', align_corners=False)
    return resampled_tensor

def generate_grid(tensor_size: tc.Tensor, device="cpu"):
    identity_transform = tc.eye(len(tensor_size)-1, device=device)[:-1, :].unsqueeze(0)
    identity_transform = tc.repeat_interleave(identity_transform, tensor_size[0], dim=0)
    grid = F.affine_grid(identity_transform, tensor_size, align_corners=False)
    return grid

def np_df_to_tc_df(displacement_field_np: np.ndarray, device: str="cpu"):
    shape = displacement_field_np.shape
    ndim = len(shape) - 1
    if ndim == 2:
        displacement_field_tc = tc.from_numpy(displacement_field_np.copy())
        displacement_field_tc = displacement_field_tc.permute(1, 2, 0).unsqueeze(0)
        temp_df_copy = displacement_field_tc.clone()
        displacement_field_tc[:, :, :, 0] = temp_df_copy[:, :, :, 0] / (shape[2]) * 2.0
        displacement_field_tc[:, :, :, 1] = temp_df_copy[:, :, :, 1] / (shape[1]) * 2.0
    if ndim == 3:
        displacement_field_tc = tc.from_numpy(displacement_field_np.copy())
        displacement_field_tc = displacement_field_tc.permute(1, 2, 3, 0).unsqueeze(0)
        temp_df_copy = displacement_field_tc.clone()
        displacement_field_tc[:, :, :, :, 0] = temp_df_copy[:, :, :, :, 2] / (shape[3]) * 2.0
        displacement_field_tc[:, :, :, :, 1] = temp_df_copy[:, :, :, :, 0] / (shape[2]) * 2.0
        displacement_field_tc[:, :, :, :, 2] = temp_df_copy[:, :, :, :, 1] / (shape[1]) * 2.0
    return displacement_field_tc.to(device)

def tc_df_to_np_df(displacement_field_tc: tc.Tensor):
    ndim = len(displacement_field_tc.size()) - 2
    if ndim == 2:
        displacement_field_np = displacement_field_tc.detach().cpu()[0].permute(2, 0, 1).numpy()
        shape = displacement_field_np.shape
        temp_df_copy = displacement_field_np.copy()
        displacement_field_np[0, :, :] = temp_df_copy[0, :, :] / 2.0 * (shape[2])
        displacement_field_np[1, :, :] = temp_df_copy[1, :, :] / 2.0 * (shape[1])
    elif ndim == 3:
        displacement_field_np = displacement_field_tc.detach().cpu()[0].permute(3, 0, 1, 2).numpy()
        shape = displacement_field_np.shape
        temp_df_copy = displacement_field_np.copy()
        displacement_field_np[0, :, :, :] = temp_df_copy[1, :, :, :] / 2.0 * (shape[2])
        displacement_field_np[1, :, :, :] = temp_df_copy[2, :, :, :] / 2.0 * (shape[1])
        displacement_field_np[2, :, :, :] = temp_df_copy[0, :, :, :] / 2.0 * (shape[3])
    return displacement_field_np

def tc_transform_to_tc_df(transformation: tc.Tensor, size: tc.Size, device: str="cpu"):
    deformation_field = F.affine_grid(transformation, size=size, align_corners=False).to(device)
    size = (deformation_field.size(0), 1) + deformation_field.size()[1:-1]
    grid = generate_grid(size, device=device)
    displacement_field = deformation_field - grid
    return displacement_field

def tc_size_to_df_size(tensor : tc.Tensor):
    tsize = tensor.size()
    ndim = len(tsize) - 2
    size = (tsize[0], ) + (tuple(list(tsize[2:],))) + (ndim,)
    return size

def resample_displacement_field(displacement_field: tc.Tensor, new_size: tc.Tensor, device: str="cpu", mode: str='bilinear'):
    sampling_grid = generate_grid((1,) + new_size[:-1]).to(device)
    resampled_displacement_field = tc.zeros(new_size).to(device)
    size = displacement_field.size()
    ndim = len(size) - 2
    for i in range(size[-1]):
        if ndim == 2:
            resampled_displacement_field[:, :, :, i] = F.grid_sample(displacement_field[:, :, :, i].unsqueeze(0), sampling_grid, mode=mode, padding_mode='zeros', align_corners=False)[0]
        elif ndim == 3:
            resampled_displacement_field[:, :, :, :, i] = F.grid_sample(displacement_field[:, :, :, :, i].unsqueeze(0), sampling_grid, mode=mode, padding_mode='zeros', align_corners=False)[0]
        else:
            raise ValueError("Unsupported number of dimensions.")
    return resampled_displacement_field

def compose_displacement_fields(displacement_field_1: tc.Tensor, displacement_field_2 : tc.Tensor, device: str="cpu"):
    size = displacement_field_1.size()
    sampling_grid = generate_grid((1,) + size[0:-1], device=device)
    composed_displacement_field = tc.zeros(size).to(device)
    ndim = len(size) - 2
    for i in range(size[-1]):
        if ndim == 2:
            pass
        elif ndim == 3:
            composed_displacement_field[:, :, :, :, i] = F.grid_sample((sampling_grid[:, :, :, :, i] + displacement_field_1[:, :, :, :, i]).unsqueeze(0), sampling_grid + displacement_field_2, padding_mode='zeros', align_corners=False)[0]
        else:
            raise ValueError("Unsupported number of dimensions.")
    composed_displacement_field = composed_displacement_field - sampling_grid
    return composed_displacement_field


def tensor_gradient(tensor: tc.Tensor, device: str="cpu"):
    ndim = len(tensor.size()) - 2
    if ndim == 2:
        gfilter_x = tc.Tensor([
            [0, 0, 0],
            [-1, 0, 1],
            [0, 0, 0],
        ]).type(tensor.type()).to(device)
        gfilter_y = tc.Tensor([
            [0, -1, 0],
            [0, 0, 0],
            [0, 1, 0],
        ]).type(tensor.type()).to(device)
        gradient_x = F.conv2d(tensor, gfilter_x.view(1, 1, 3, 3), padding=1) / 2.0
        gradient_y = F.conv2d(tensor, gfilter_y.view(1, 1, 3, 3), padding=1) / 2.0
        return gradient_y, gradient_x
    elif ndim == 3:
        gfilter_z = tc.Tensor([
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            [
                [0, 0, 0],
                [-1, 0, 1],
                [0, 0, 0],
            ],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
        ]).type(tensor.type()).to(device)
        gfilter_x = tc.Tensor([
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            [
                [0, -1, 0],
                [0, 0, 0],
                [0, 1, 0],
            ],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
        ]).type(tensor.type()).to(device)
        gfilter_y = tc.Tensor([
            [
                [0, 0, 0],
                [0, -1, 0],
                [0, 0, 0],
            ],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
        ]).type(tensor.type()).to(device)
        gradient_x = F.conv3d(tensor, gfilter_x.view(1, 1, 3, 3, 3), padding=1) / 2.0
        gradient_y = F.conv3d(tensor, gfilter_y.view(1, 1, 3, 3, 3), padding=1) / 2.0
        gradient_z = F.conv3d(tensor, gfilter_z.view(1, 1, 3, 3, 3), padding=1) / 2.0
        return gradient_y, gradient_x, gradient_z
    else:
        raise ValueError("Unsupported number of dimensions.")

def tensor_laplacian(tensor: tc.Tensor, device: str="cpu"):
    ndim = len(tensor.size()) - 2
    if ndim == 2:
        lfilter = tc.Tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0],
        ]).type(tensor.type()).to(device)
        laplacian = F.conv2d(tensor, lfilter.view(1, 1, 3, 3), padding=1)
    elif ndim == 3:
        lfilter = tc.Tensor([
            [
                [0, 0, 0],
                [0, -1, 0],
                [0, 0, 0],
            ],
            [
                [0, -1, 0],
                [-1, 6, -1],
                [0, -1, 0],
            ],
            [
                [0, 0, 0],
                [0, -1, 0],
                [0, 0, 0],
            ],
        ]).type(tensor.type()).to(device)
        laplacian = F.conv3d(tensor, lfilter.view(1, 1, 3, 3, 3), padding=1)
    else:
        raise ValueError("Unsupported number of dimensions.")
    return laplacian

def create_pyramid(tensor: tc.Tensor, num_levels: int, device: str="cpu", mode: str='bilinear'):
    pyramid = []
    for i in range(num_levels):
        if i == num_levels - 1:
            pyramid.append(tensor)
        else:
            current_size = tensor.size()
            new_size = (int(current_size[j]/(2**(num_levels-i-1))) if j > 1 else current_size[j] for j in range(len(current_size)))
            new_size = tc.Size(new_size)
            new_tensor = resample_tensor(tensor, new_size, device=device, mode=mode)
            pyramid.append(new_tensor)
    return pyramid