import os

import numpy as np
import torch as tc

import instance_optimization as io
import cost_functions as cf
import regularizers as rg
import utils as u
import utils_tc as utc
from networks import nonrigid_network as nn
from networks import nonrigid_network_multilevel as nnm


def affine_nonrigid_registration(source, target, params):
    try:
        num_levels = params['num_levels']
        used_levels = params['used_levels']
        num_iters = params['num_iters']
        learning_rate = params['learning_rate']
        cost_function = params['cost_function']
        cost_function_params = params['cost_function_params']
        nonrigid_model = params['nonrigid_model']
        device = params['device']
    except:
        num_levels = 3
        used_levels = 1
        num_iters = 400
        learning_rate = 0.01
        cost_function = cf.ncc_local_tc
        cost_function_params = {}
        nonrigid_model = None # TO DO
        device = tc.device("cuda:0" if tc.cuda.is_available() else "cpu")

    model = nn.load_network(device, nonrigid_model)
    model.eval()

    spacing = params['spacing']
    new_spacing = (2.0, 2.0, 2.0)
    early_iters = num_iters

    source_resampled = u.resample_to_reg(source, spacing, new_spacing)
    target_resampled = u.resample_to_reg(target, spacing, new_spacing)

    print("Source shape: ", source_resampled.shape)

    source_tensor = tc.from_numpy(source_resampled.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0)
    target_tensor = tc.from_numpy(target_resampled.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0)
    transformation = io.affine_registration(source_tensor, target_tensor, num_levels, used_levels, num_iters, learning_rate, cost_function, cost_function_params, early_iters, device=device)
    df_affine = utc.tc_transform_to_tc_df(transformation, source_tensor.size(), device=device)
    with tc.set_grad_enabled(False):
        warped_source_tensor = utc.warp_tensor(source_tensor, df_affine, device=device)
        source_normalized = (warped_source_tensor - tc.min(warped_source_tensor)) / (tc.max(warped_source_tensor) - tc.min(warped_source_tensor))
        target_normalized = (target_tensor - tc.min(target_tensor)) / (tc.max(target_tensor) - tc.min(target_tensor))
        df_nonrigid = model(source_normalized, target_normalized)
        df_final = utc.compose_displacement_fields(df_affine, df_nonrigid, device=device)
    print("Initial loss: ", cost_function(source_tensor, target_tensor, device=device))
    print("Affine loss: ", cost_function(warped_source_tensor, target_tensor, device=device))
    print("Affine/Nonrigid loss: ", cost_function(utc.warp_tensor(source_tensor, df_final, device=device), target_tensor, device=device))
    df_final = utc.resample_displacement_field(df_final, (1, ) + source.shape + (3, ), device=device)
    df_np_final = utc.tc_df_to_np_df(df_final)
    u_x, u_y, u_z = df_np_final[0, :, :, :], df_np_final[1, :, :, :], df_np_final[2, :, :, :]
    return u_x, u_y, u_z

def affine_nonrigid_penalty_registration(source, target, params):
    try:
        num_levels = params['num_levels']
        used_levels = params['used_levels']
        num_iters = params['num_iters']
        learning_rate = params['learning_rate']
        cost_function = params['cost_function']
        cost_function_params = params['cost_function_params']
        nonrigid_model = params['nonrigid_model']
        device = params['device']
    except:
        num_levels = 3
        used_levels = 1
        num_iters = 400
        learning_rate = 0.01
        cost_function = cf.ncc_local_tc
        cost_function_params = {}
        nonrigid_model = None # TO DO
        device = tc.device("cuda:0" if tc.cuda.is_available() else "cpu")

    model = nn.load_network(device, nonrigid_model)
    model.eval()

    spacing = params['spacing']
    new_spacing = (2.0, 2.0, 2.0)
    early_iters = num_iters

    source_resampled = u.resample_to_reg(source, spacing, new_spacing)
    target_resampled = u.resample_to_reg(target, spacing, new_spacing)

    print("Source shape: ", source_resampled.shape)

    source_tensor = tc.from_numpy(source_resampled.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0)
    target_tensor = tc.from_numpy(target_resampled.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0)
    transformation = io.affine_registration(source_tensor, target_tensor, num_levels, used_levels, num_iters, learning_rate, cost_function, cost_function_params, early_iters, device=device)
    df_affine = utc.tc_transform_to_tc_df(transformation, source_tensor.size(), device=device)
    with tc.set_grad_enabled(False):
        warped_source_tensor = utc.warp_tensor(source_tensor, df_affine, device=device)
        source_normalized = (warped_source_tensor - tc.min(warped_source_tensor)) / (tc.max(warped_source_tensor) - tc.min(warped_source_tensor))
        target_normalized = (target_tensor - tc.min(target_tensor)) / (tc.max(target_tensor) - tc.min(target_tensor))
        df_nonrigid = model(source_normalized, target_normalized)
        df_final = utc.compose_displacement_fields(df_affine, df_nonrigid, device=device)
    print("Initial loss: ", cost_function(source_tensor, target_tensor, device=device))
    print("Affine loss: ", cost_function(warped_source_tensor, target_tensor, device=device))
    print("Affine/Nonrigid loss: ", cost_function(utc.warp_tensor(source_tensor, df_final, device=device), target_tensor, device=device))
    df_final = utc.resample_displacement_field(df_final, (1, ) + source.shape + (3, ), device=device)
    df_np_final = utc.tc_df_to_np_df(df_final)
    u_x, u_y, u_z = df_np_final[0, :, :, :], df_np_final[1, :, :, :], df_np_final[2, :, :, :]
    return u_x, u_y, u_z


def affine_nonrigid_multilevel_registration(source, target, params):
    try:
        num_levels = params['num_levels']
        used_levels = params['used_levels']
        num_iters = params['num_iters']
        learning_rate = params['learning_rate']
        cost_function = params['cost_function']
        cost_function_params = params['cost_function_params']
        nonrigid_model = params['nonrigid_model']
        device = params['device']
    except:
        num_levels = 3
        used_levels = 1
        num_iters = 400
        learning_rate = 0.01
        cost_function = cf.ncc_local_tc
        cost_function_params = {}
        nonrigid_model = None # TO DO
        device = tc.device("cuda:0" if tc.cuda.is_available() else "cpu")

    model = nnm.load_network(device, nonrigid_model)
    model.eval()

    spacing = params['spacing']
    new_spacing = (2.0, 2.0, 2.0)
    early_iters = num_iters

    source_resampled = u.resample_to_reg(source, spacing, new_spacing)
    target_resampled = u.resample_to_reg(target, spacing, new_spacing)

    print("Source shape: ", source_resampled.shape)

    source_tensor = tc.from_numpy(source_resampled.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0)
    target_tensor = tc.from_numpy(target_resampled.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0)
    transformation = io.affine_registration(source_tensor, target_tensor, num_levels, used_levels, num_iters, learning_rate, cost_function, cost_function_params, early_iters, device=device)
    df_affine = utc.tc_transform_to_tc_df(transformation, source_tensor.size(), device=device)
    with tc.set_grad_enabled(False):
        warped_source_tensor = utc.warp_tensor(source_tensor, df_affine, device=device)
        source_normalized = (warped_source_tensor - tc.min(warped_source_tensor)) / (tc.max(warped_source_tensor) - tc.min(warped_source_tensor))
        target_normalized = (target_tensor - tc.min(target_tensor)) / (tc.max(target_tensor) - tc.min(target_tensor))
        source_pyramid = utc.create_pyramid(source_normalized, 3, device=device)
        target_pyramid = utc.create_pyramid(target_normalized, 3, device=device)
        df_nonrigid = model(source_pyramid, target_pyramid)[-1]
        df_final = utc.compose_displacement_fields(df_affine, df_nonrigid, device=device)
    print("Initial loss: ", cost_function(source_tensor, target_tensor, device=device))
    print("Affine loss: ", cost_function(warped_source_tensor, target_tensor, device=device))
    print("Affine/Nonrigid loss: ", cost_function(utc.warp_tensor(source_tensor, df_final, device=device), target_tensor, device=device))
    df_final = utc.resample_displacement_field(df_final, (1, ) + source.shape + (3, ), device=device)
    df_np_final = utc.tc_df_to_np_df(df_final)
    u_x, u_y, u_z = df_np_final[0, :, :, :], df_np_final[1, :, :, :], df_np_final[2, :, :, :]
    return u_x, u_y, u_z

def affine_nonrigid_symmetric_registration(source, target, params):
    try:
        num_levels = params['num_levels']
        used_levels = params['used_levels']
        num_iters = params['num_iters']
        learning_rate = params['learning_rate']
        cost_function = params['cost_function']
        cost_function_params = params['cost_function_params']
        nonrigid_model = params['nonrigid_model']
        device = params['device']
    except:
        num_levels = 3
        used_levels = 1
        num_iters = 400
        learning_rate = 0.01
        cost_function = cf.ncc_local_tc
        cost_function_params = {}
        nonrigid_model = None # TO DO
        device = tc.device("cuda:0" if tc.cuda.is_available() else "cpu")

    model = nn.load_network(device, nonrigid_model)
    model.eval()

    spacing = params['spacing']
    new_spacing = (2.0, 2.0, 2.0)
    early_iters = num_iters

    source_resampled = u.resample_to_reg(source, spacing, new_spacing)
    target_resampled = u.resample_to_reg(target, spacing, new_spacing)

    print("Source shape: ", source_resampled.shape)

    source_tensor = tc.from_numpy(source_resampled.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0)
    target_tensor = tc.from_numpy(target_resampled.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0)
    transformation = io.affine_registration(source_tensor, target_tensor, num_levels, used_levels, num_iters, learning_rate, cost_function, cost_function_params, early_iters, device=device)
    df_affine = utc.tc_transform_to_tc_df(transformation, source_tensor.size(), device=device)
    with tc.set_grad_enabled(False):
        warped_source_tensor = utc.warp_tensor(source_tensor, df_affine, device=device)
        source_normalized = (warped_source_tensor - tc.min(warped_source_tensor)) / (tc.max(warped_source_tensor) - tc.min(warped_source_tensor))
        target_normalized = (target_tensor - tc.min(target_tensor)) / (tc.max(target_tensor) - tc.min(target_tensor))
        df_nonrigid = model(source_normalized, target_normalized)
        df_final = utc.compose_displacement_fields(df_affine, df_nonrigid, device=device)
    print("Initial loss: ", cost_function(source_tensor, target_tensor, device=device))
    print("Affine loss: ", cost_function(warped_source_tensor, target_tensor, device=device))
    print("Affine/Nonrigid loss: ", cost_function(utc.warp_tensor(source_tensor, df_final, device=device), target_tensor, device=device))
    df_final = utc.resample_displacement_field(df_final, (1, ) + source.shape + (3, ), device=device)
    df_np_final = utc.tc_df_to_np_df(df_final)
    u_x, u_y, u_z = df_np_final[0, :, :, :], df_np_final[1, :, :, :], df_np_final[2, :, :, :]
    return u_x, u_y, u_z

def affine_nonrigid_multilevel_penalty_registration(source, target, params):
    try:
        num_levels = params['num_levels']
        used_levels = params['used_levels']
        num_iters = params['num_iters']
        learning_rate = params['learning_rate']
        cost_function = params['cost_function']
        cost_function_params = params['cost_function_params']
        nonrigid_model = params['nonrigid_model']
        device = params['device']
    except:
        num_levels = 3
        used_levels = 1
        num_iters = 400
        learning_rate = 0.01
        cost_function = cf.ncc_local_tc
        cost_function_params = {}
        nonrigid_model = None # TO DO
        device = tc.device("cuda:0" if tc.cuda.is_available() else "cpu")

    model = nnm.load_network(device, nonrigid_model)
    model.eval()

    spacing = params['spacing']
    new_spacing = (2.0, 2.0, 2.0)
    early_iters = num_iters

    source_resampled = u.resample_to_reg(source, spacing, new_spacing)
    target_resampled = u.resample_to_reg(target, spacing, new_spacing)

    print("Source shape: ", source_resampled.shape)

    source_tensor = tc.from_numpy(source_resampled.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0)
    target_tensor = tc.from_numpy(target_resampled.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0)
    transformation = io.affine_registration(source_tensor, target_tensor, num_levels, used_levels, num_iters, learning_rate, cost_function, cost_function_params, early_iters, device=device)
    df_affine = utc.tc_transform_to_tc_df(transformation, source_tensor.size(), device=device)
    with tc.set_grad_enabled(False):
        warped_source_tensor = utc.warp_tensor(source_tensor, df_affine, device=device)
        source_normalized = (warped_source_tensor - tc.min(warped_source_tensor)) / (tc.max(warped_source_tensor) - tc.min(warped_source_tensor))
        target_normalized = (target_tensor - tc.min(target_tensor)) / (tc.max(target_tensor) - tc.min(target_tensor))
        source_pyramid = utc.create_pyramid(source_normalized, 3, device=device)
        target_pyramid = utc.create_pyramid(target_normalized, 3, device=device)
        df_nonrigid = model(source_pyramid, target_pyramid)[-1]
        df_final = utc.compose_displacement_fields(df_affine, df_nonrigid, device=device)
    print("Initial loss: ", cost_function(source_tensor, target_tensor, device=device))
    print("Affine loss: ", cost_function(warped_source_tensor, target_tensor, device=device))
    print("Affine/Nonrigid loss: ", cost_function(utc.warp_tensor(source_tensor, df_final, device=device), target_tensor, device=device))
    df_final = utc.resample_displacement_field(df_final, (1, ) + source.shape + (3, ), device=device)
    df_np_final = utc.tc_df_to_np_df(df_final)
    u_x, u_y, u_z = df_np_final[0, :, :, :], df_np_final[1, :, :, :], df_np_final[2, :, :, :]
    return u_x, u_y, u_z
