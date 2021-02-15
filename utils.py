import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import ndimage as nd
import SimpleITK as sitk


def load_case(data_path, case_id):
    case_path = os.path.join(data_path, str(case_id))
    source_path = os.path.join(case_path, "source.mha")
    target_path = os.path.join(case_path, "target.mha")
    source_mask_path = os.path.join(case_path, "source_segmentation.mha")
    source_landmarks_path = os.path.join(case_path, "source_landmarks.csv")
    target_landmarks_path = os.path.join(case_path, "target_landmarks.csv")
    source_landmarks_md_path = os.path.join(case_path, "source_landmarks_md.csv")
    target_landmarks_md_path = os.path.join(case_path, "target_landmarks_md.csv")

    source, _ = load_volume(source_path)
    target, spacing = load_volume(target_path)
    source_mask = load_segmentation(source_mask_path)
    source_landmarks = load_landmarks(source_landmarks_path)
    target_landmarks = load_landmarks(target_landmarks_path)
    source_landmarks_md = load_landmarks(source_landmarks_md_path)
    target_landmarks_md = load_landmarks(target_landmarks_md_path)

    return source, target, source_mask, spacing, source_landmarks, target_landmarks, source_landmarks_md, target_landmarks_md

def save_case(source, target, source_mask, spacing, source_landmarks, target_landmarks, source_landmarks_md, target_landmarks_md, data_path, case_id):
    case_path = os.path.join(data_path, str(case_id))
    if not os.path.isdir(case_path):
        os.makedirs(case_path)
    save_volume(os.path.join(case_path, "source.mha"), source, spacing)
    save_volume(os.path.join(case_path, "target.mha"), target, spacing)
    save_segmentation(os.path.join(case_path, "source_segmentation.mha"), source_mask)
    save_landmarks(os.path.join(case_path, "source_landmarks.csv"), source_landmarks)
    save_landmarks(os.path.join(case_path, "target_landmarks.csv"), target_landmarks)
    save_landmarks(os.path.join(case_path, "source_landmarks_md.csv"), source_landmarks_md)
    save_landmarks(os.path.join(case_path, "target_landmarks_md.csv"), target_landmarks_md)

def load_volume(input_path):
    volume_image = sitk.ReadImage(input_path)
    spacing = volume_image.GetSpacing()
    volume = sitk.GetArrayFromImage(volume_image).astype(np.float32).swapaxes(0, 2).swapaxes(0, 1)
    return volume, spacing

def load_segmentation(input_path):
    segmentation_volume_image = sitk.ReadImage(input_path)
    segmentation_volume = sitk.GetArrayFromImage(segmentation_volume_image).swapaxes(0, 2).swapaxes(0, 1)
    return segmentation_volume

def load_landmarks(input_path):
    landmarks = pd.read_csv(input_path).to_numpy()[:, 1:]
    return landmarks

def save_volume(save_path, volume, spacing):
    volume_image = sitk.GetImageFromArray(volume.astype(np.float32).swapaxes(0, 1).swapaxes(0, 2))
    volume_image.SetSpacing(spacing)
    sitk.WriteImage(volume_image, save_path)

def save_segmentation(save_path, volume):
    segmentation_volume_image = sitk.GetImageFromArray(volume.astype(np.uint8).swapaxes(0, 1).swapaxes(0, 2))
    sitk.WriteImage(segmentation_volume_image, save_path)

def save_landmarks(save_path, landmarks):
    df = pd.DataFrame(landmarks, columns=['X', 'Y', 'Z'])
    df.to_csv(save_path)

def warp_landmarks(landmarks, displacement_field):
    landmarks = landmarks.copy()
    landmarks_x = landmarks[:, 0]
    landmarks_y = landmarks[:, 1]
    landmarks_z = landmarks[:, 2]
    u_x = displacement_field[0, :, :, :]
    u_y = displacement_field[1, :, :, :]
    u_z = displacement_field[2, :, :, :]
    ux = ndimage.map_coordinates(u_x, [landmarks_y, landmarks_x, landmarks_z], mode='nearest')
    uy = ndimage.map_coordinates(u_y, [landmarks_y, landmarks_x, landmarks_z], mode='nearest')
    uz = ndimage.map_coordinates(u_z, [landmarks_y, landmarks_x, landmarks_z], mode='nearest')
    new_landmarks = np.stack((landmarks_x + ux, landmarks_y + uy, landmarks_z + uz), axis=1)
    return new_landmarks

def warp_landmarks_2(landmarks, u_x, u_y, u_z):
    landmarks = landmarks.copy()
    landmarks_x = landmarks[:, 0]
    landmarks_y = landmarks[:, 1]
    landmarks_z = landmarks[:, 2]
    ux = nd.map_coordinates(u_x, [landmarks_y, landmarks_x, landmarks_z], mode='nearest')
    uy = nd.map_coordinates(u_y, [landmarks_y, landmarks_x, landmarks_z], mode='nearest')
    uz = nd.map_coordinates(u_z, [landmarks_y, landmarks_x, landmarks_z], mode='nearest')
    new_landmarks = np.stack((landmarks_x + ux, landmarks_y + uy, landmarks_z + uz), axis=1)
    return new_landmarks

def compose_vector_fields(u_x, u_y, u_z, v_x, v_y, v_z):
    y_size, x_size, z_size = np.shape(u_x)
    grid_x, grid_y, grid_z = np.meshgrid(np.arange(x_size), np.arange(y_size), np.arange(z_size))
    added_y = grid_y + v_y
    added_x = grid_x + v_x
    added_z = grid_z + v_z
    t_x = nd.map_coordinates(grid_x + u_x, [added_y, added_x, added_z], mode='constant', cval=0.0)
    t_y = nd.map_coordinates(grid_y + u_y, [added_y, added_x, added_z], mode='constant', cval=0.0)
    t_z = nd.map_coordinates(grid_z + u_z, [added_y, added_x, added_z], mode='constant', cval=0.0)
    n_x, n_y, n_z = t_x - grid_x, t_y - grid_y, t_z - grid_z
    indexes_x = np.logical_or(added_x >= x_size - 1, added_x <= 0)
    indexes_y = np.logical_or(added_y >= y_size - 1, added_y <= 0)
    indexes_z = np.logical_or(added_z >= z_size - 1, added_z <= 0)
    indexes = np.logical_or(np.logical_or(indexes_x, indexes_y), indexes_z)
    n_x[indexes] = 0.0
    n_y[indexes] = 0.0
    n_z[indexes] = 0.0
    return n_x, n_y, n_z

def resample(image, output_x_size, output_y_size, output_z_size, order=3):
    y_size, x_size, z_size = np.shape(image)
    out_grid_x, out_grid_y, out_grid_z = np.meshgrid(np.arange(output_x_size), np.arange(output_y_size), np.arange(output_z_size))
    out_grid_x = out_grid_x * x_size / output_x_size
    out_grid_y = out_grid_y * y_size / output_y_size
    out_grid_z = out_grid_z * z_size / output_z_size
    image = ndimage.map_coordinates(image, [out_grid_y, out_grid_x, out_grid_z], order=order, cval=0.0)
    return image

def resample_to_reg(image, old_spacing, new_spacing, order=3):
    y_size, x_size, z_size = np.shape(image)
    image = resample(image, int(x_size * old_spacing[0] / new_spacing[0]), int(y_size * old_spacing[1] / new_spacing[1]), int(z_size  * old_spacing[2] / new_spacing[2]), order=order)
    return image

def warp_volume(volume, u_x, u_y, u_z):
    result = warp.backward_warping(volume, u_x, u_y, u_z, order=3)
    return result

def warp_segmentation(segmentation, u_x, u_y, u_z):
    result = np.zeros(segmentation.shape).astype(np.uint8)
    no_uniques = len(np.unique(segmentation))
    for i in range(1, no_uniques):
        temp_result = (warp.backward_warping((segmentation == i).astype(np.float), u_x, u_y, u_z, order=3) > 0.5).astype(np.uint8)
        result[temp_result == 1] = i
    return result

def segmentation_volume(segmentation, spacing, mask_id=1):
    pixel_size = spacing[0]*spacing[1]*spacing[2]
    total_count = np.count_nonzero(segmentation == mask_id)
    return total_count*pixel_size

def inverse_consistency(u_x, u_y, u_z, inv_u_x, inv_u_y, inv_u_z):
    y_size, x_size, z_size = u_x.shape
    n_u_x, n_u_y, n_u_z = compose_vector_fields(u_x, u_y, u_z, inv_u_x, inv_u_y, inv_u_z)
    ic = np.sqrt(np.square(n_u_x) + np.square(n_u_y) + np.square(n_u_z))
    return ic

def mask_volume(mask : np.ndarray):
    return np.sum(mask).astype(np.float32)

def points_to_homogeneous_representation(points: np.ndarray):
    homogenous_points = np.concatenate((points, np.ones((points.shape[0], 1), dtype=points.dtype)), axis=1)
    return homogenous_points

def matrix_transform(points: np.ndarray, matrix: np.ndarray):
    points = points_to_homogeneous_representation(points)
    transformed_points = (points @ matrix.T)[:, 0:-1]
    return transformed_points

def tre(source_points: np.ndarray, target_points: np.ndarray, spacing: np.ndarray=None):
    if spacing is None or len(spacing) != source_points.shape[1]:
        spacing = np.array([1.0] * source_points.shape[1], dtype=source_points.dtype)
    source_points = source_points*spacing
    target_points = target_points*spacing
    distances = np.sqrt(np.sum((source_points - target_points)**2, axis=1))
    return distances

def move_matrix(matrix: np.ndarray, origin: np.ndarray):
    if len(origin) != matrix.shape[0] - 1:
        raise ValueError("Unsupported matrix dimension.")
    origin = np.array(origin, dtype=matrix.dtype)
    lm = np.eye(matrix.shape[0])
    rm = np.eye(matrix.shape[0])
    lm[:-1,-1] = origin
    rm[:-1,-1] = -origin
    matrix = lm @ matrix @ rm
    return matrix

def image_matrix_warping(image: np.ndarray, matrix: np.ndarray, order: int=1, cval: float=0.0, origin: tuple=None):
    dims = len(image.shape)
    if origin is not None:
        if len(origin) != dims:
            raise ValueError("Incorrect origin.")
        matrix = move_matrix(matrix, origin)
    if dims == 2:
        grid_x, grid_y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        points = np.stack((grid_x.ravel(), grid_y.ravel())).T
        transformed_points = matrix_transform(points, matrix)
        displacement_field = np.zeros((2, image.shape[0], image.shape[1]))
        displacement_field[0, :, :] = transformed_points[:, 0].reshape(image.shape) - grid_x
        displacement_field[1, :, :] = transformed_points[:, 1].reshape(image.shape) - grid_y
    elif dims == 3:
        grid_x, grid_y, grid_z = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]), np.arange(image.shape[2]))
        points = np.stack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel())).T
        transformed_points = matrix_transform(points, matrix).T
        displacement_field = np.zeros((3, image.shape[0], image.shape[1], image.shape[2]))
        displacement_field[0, :, :, :] = transformed_points[0, :].reshape(image.shape) - grid_x
        displacement_field[1, :, :, :] = transformed_points[1, :].reshape(image.shape) - grid_y
        displacement_field[2, :, :, :] = transformed_points[2, :].reshape(image.shape) - grid_z
    else:
        raise ValueError("Unsupported number of dimensions.")    
    return image_warping(image, displacement_field, order=order, cval=cval)

def image_warping(image: np.ndarray, displacement_field: np.ndarray, order: int=1, cval: float=0.0):
    dims = len(image.shape)
    if dims == 2:
        grid_x, grid_y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        transformed_image = ndimage.map_coordinates(image, [grid_y + displacement_field[1], grid_x + displacement_field[0]], order=order, cval=cval)
    elif dims == 3:
        grid_x, grid_y, grid_z = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]), np.arange(image.shape[2]))
        transformed_image = ndimage.map_coordinates(image, [grid_y + displacement_field[1], grid_x + displacement_field[0], grid_z + displacement_field[2]], order=order, cval=cval)
    else:
        raise ValueError("Unsupported number of dimensions.")
    return transformed_image

def transform_points_to_physicial_spacing(points: np.ndarray, spacing: np.ndarray):
    spacing = np.array(spacing, dtype=points.dtype)
    transformed_points = points * spacing
    return transformed_points

def transform_matrix_to_image_spacing(matrix: np.ndarray, spacing: np.ndarray):
    transformed_matrix = matrix.copy()
    spacing = np.array(spacing, dtype=matrix.dtype)
    transformed_matrix[:-1, -1] = matrix[:-1, -1] / spacing
    return transformed_matrix

def resample_to_spacing(image: np.ndarray, old_spacing: np.ndarray, new_spacing: np.ndarray, order: int=1, cval: float=0.0):
    dims = len(image.shape)
    if dims == 2:
        grid_x, grid_y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        grid_x = grid_x / (old_spacing[0] / new_spacing[0])
        grid_y = grid_y / (old_spacing[1] / new_spacing[1])
        transformed_image = ndimage.map_coordinates(image, [grid_y, grid_x], order=order, cval=cval)
    elif dims == 3:
        grid_x, grid_y, grid_z = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]), np.arange(image.shape[2]))
        grid_x = grid_x / (old_spacing[0] / new_spacing[0])
        grid_y = grid_y / (old_spacing[1] / new_spacing[1])
        grid_z = grid_z / (old_spacing[2] / new_spacing[2])
        transformed_image = ndimage.map_coordinates(image, [grid_y, grid_x, grid_z], order=order, cval=cval)
    else:
        raise ValueError("Unsupported number of dimensions.")
    return transformed_image

def resample_to_shape(image: np.ndarray, new_shape: np.ndarray, order: int=1, cval: float=0.0):
    shape = image.shape
    dims = len(shape)
    if dims == 2:
        grid_x, grid_y = np.meshgrid(np.arange(new_shape[1]), np.arange(new_shape[0]))
        grid_x = grid_x * (shape[1] / new_shape[1])
        grid_y = grid_y * (shape[0] / new_shape[0])
        transformed_image = ndimage.map_coordinates(image, [grid_y, grid_x], order=order, cval=cval)
    elif dims == 3:
        grid_x, grid_y, grid_z = np.meshgrid(np.arange(new_shape[1]), np.arange(new_shape[0]), np.arange(new_shape[2]))
        grid_x = grid_x * (shape[1] / new_shape[1])
        grid_y = grid_y * (shape[0] / new_shape[0])
        grid_z = grid_z * (shape[2] / new_shape[2])
        transformed_image = ndimage.map_coordinates(image, [grid_y, grid_x, grid_z], order=order, cval=cval)
    else:
        raise ValueError("Unsupported number of dimensions.")
    return transformed_image

def resample_to_spacing_by_resolution(image, old_spacing, new_spacing, order: int=1, cval: float=0.0):
    shape = image.shape
    multiplier = (np.array(old_spacing, dtype=np.float32) / np.array(new_spacing, dtype=np.float32))
    multiplier[0], multiplier[1] = multiplier[1], multiplier[0] # Swap x,y
    new_shape = shape * multiplier
    new_shape = np.ceil(new_shape).astype(np.int)
    transformed_image = resample_to_shape(image, new_shape, order=order, cval=cval) 
    return transformed_image

def pad_to_given_shape(image: np.ndarray, new_shape: np.ndarray, cval: float=0.0):
    shape = image.shape
    diff = np.array(new_shape) - np.array(shape)
    diff = np.maximum(diff, 0)
    diff_l = np.floor(diff / 2).astype(np.int)
    diff_r = np.ceil(diff / 2).astype(np.int)
    padded_image = np.pad(image, np.array([diff_l, diff_r]).T, constant_values=cval)
    return padded_image

def pad_to_same_shape(image_1: np.ndarray, image_2: np.ndarray, cval: float=0.0):
    shape_1 = image_1.shape
    shape_2 = image_2.shape
    new_shape = np.maximum(shape_1, shape_2)
    padded_image_1 = pad_to_given_shape(image_1, new_shape, cval=cval)
    padded_image_2 = pad_to_given_shape(image_2, new_shape, cval=cval)
    return padded_image_1, padded_image_2

def pad_and_resample(image_1: np.ndarray, image_2: np.ndarray, image_1_spacing: np.ndarray, image_2_spacing: np.ndarray,
    mode: str="max", order: int=1, cval: float=0.0):
    if mode == "max":
        new_spacing = np.maximum(np.array(image_1_spacing), np.array(image_2_spacing))
    elif mode == "min":
        new_spacing = np.minimum(np.array(image_1_spacing), np.array(image_2_spacing))
    else:
        raise ValueError("Unsupported spacing calculation mode.")
    resampled_image_1 = resample_to_spacing_by_resolution(image_1, image_1_spacing, new_spacing, order=order, cval=cval)
    resampled_image_2 = resample_to_spacing_by_resolution(image_2, image_2_spacing, new_spacing, order=order, cval=cval)
    padded_image_1, padded_image_2 = pad_to_same_shape(resampled_image_1, resampled_image_2, cval=cval)
    return padded_image_1, padded_image_2, new_spacing

def normalize(image: np.ndarray):
    return (image - np.min(image)) / (np.max(image) - np.min(image))

def normalize_to_window(image: np.ndarray, vmin: float, vmax: float):
    normalized_image = image.copy()
    normalized_image[normalized_image < vmin] = vmin
    normalized_image[normalized_image > vmax] = vmax
    normalized_image = (normalized_image - vmin) / (vmax - vmin)
    return normalized_image 

def checkerboard_image(image_1: np.ndarray, image_2: np.ndarray, step: int=20):
    checkerboard = np.empty(image_1.shape)
    ndim = len(image_1.shape)
    if ndim == 2:
        t = True
        y_size, x_size = image_1.shape
        for i in range(0, x_size, step):
            for j in range(0, y_size, step):
                b_x = max(0, i)
                b_y = max(0, j)
                e_x = min(x_size, i+step)
                e_y = min(y_size, j+step)
                if t:
                    checkerboard[b_y:e_y, b_x:e_x] = image_1[b_y:e_y, b_x:e_x]
                else:
                    checkerboard[b_y:e_y, b_x:e_x] = image_2[b_y:e_y, b_x:e_x]
                t = not t
            if len(np.arange(0, y_size, step)) % 2 == 0:
                t = not t
    elif ndim == 3:
        t = True
        y_size, x_size, z_size = image_1.shape
        for k in range(0, z_size, step):
            for i in range(0, x_size, step):
                for j in range(0, y_size, step):
                    b_x = max(0, i)
                    b_y = max(0, j)
                    b_z = max(0, k)
                    e_x = min(x_size, i+step)
                    e_y = min(y_size, j+step)
                    e_z = min(z_size, k+step)
                    if t:
                        checkerboard[b_y:e_y, b_x:e_x, b_z:e_z] = image_1[b_y:e_y, b_x:e_x, b_z:e_z]
                    else:
                        checkerboard[b_y:e_y, b_x:e_x, b_z:e_z] = image_2[b_y:e_y, b_x:e_x, b_z:e_z]
                    t = not t
                if len(np.arange(0, y_size, step)) % 2 == 0:
                    t = not t
    else:
        raise ValueError("Unsupported dimension.")
    return checkerboard
