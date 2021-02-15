import os
import time

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import math

import torch as tc
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.nn.functional as F

import dataloaders as dl
import cost_functions as cf
import regularizers as rg
import utils
import utils_tc as utc

from networks import nonrigid_network as nn

data_path = None # TO DO
models_path = None # TO DO
figures_path = None # TO DO
device = tc.device("cuda:0" if torch.cuda.is_available() else "cpu")


def training(training_params):
    model_name = training_params['model_name']
    num_epochs = training_params['num_epochs']
    batch_size = training_params['batch_size']
    learning_rate = training_params['learning_rate'] 
    initial_path = training_params['initial_path']
    decay_rate = training_params['decay_rate']
    alpha = training_params['alpha']
    transforms = training_params['transforms']
    model_save_path = os.path.join(models_path, model_name)

    model = nn.load_network(device, path=initial_path).to(device)
    parameters = model.parameters()
    optimizer = optim.Adam(parameters, learning_rate)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: decay_rate**epoch)

    training_ids = None # TO DO
    validation_ids = None # TO DO
    training_loader = dl.UnsupervisedLoader(data_path, training_ids, transforms=transforms)
    validation_loader = dl.UnsupervisedLoader(data_path, validation_ids, transforms=None) 
    training_dataloader = tc.utils.data.DataLoader(training_loader, batch_size = batch_size, shuffle = True, num_workers = 4, collate_fn = dl.collate_to_list_unsupervised)
    validation_dataloader = tc.utils.data.DataLoader(validation_loader, batch_size = batch_size, shuffle = True, num_workers = 4, collate_fn = dl.collate_to_list_unsupervised)

    cost_function = cf.ncc_local_tc
    cost_function_params = {'win_size': 5}

    reg_function = rg.diffusion_tc
    reg_function_params = dict()

    # Training starts here
    train_history = []
    val_history = []
    train_reg_history = []
    val_reg_history = []
    train_tre_history = []
    val_tre_history = []
    train_tre_md_history = []
    val_tre_md_history = []
    train_ic_history = []
    val_ic_history = []
    training_size = len(training_dataloader.dataset)
    validation_size = len(validation_dataloader.dataset)
    print("Training size: ", training_size)
    print("Validation size: ", validation_size)

    initial_training_loss = 0.0
    initial_validation_loss = 0.0
    initial_training_tre = 0.0
    initial_validation_tre = 0.0
    initial_training_md_tre = 0.0
    initial_validation_md_tre = 0.0
    for sources, targets, sources_masks, spacings, sources_landmarks, targets_landmarks, sources_landmarks_md, targets_landmarks_md in training_dataloader:
        with torch.set_grad_enabled(False):
            for i in range(len(sources)):
                source = sources[i]
                target = targets[i]
                source_landmarks = sources_landmarks[i]
                target_landmarks = targets_landmarks[i]
                source_landmarks_md = sources_landmarks_md[i]
                target_landmarks_md = targets_landmarks_md[i]
                spacing = spacings[i]
                source = source.to(device).view(1, 1, source.size(0), source.size(1), source.size(2))
                target = target.to(device).view(1, 1, target.size(0), target.size(1), target.size(2))
                loss = cost_function(source, target, device=device, **cost_function_params)
                initial_training_loss += loss.item()
                initial_training_tre += np.mean(utils.tre(source_landmarks, target_landmarks, spacing))
                initial_training_md_tre += np.mean(utils.tre(source_landmarks_md, target_landmarks_md, spacing))
    for sources, targets, sources_masks, spacings, sources_landmarks, targets_landmarks, sources_landmarks_md, targets_landmarks_md in validation_dataloader:
        with torch.set_grad_enabled(False):
            for i in range(len(sources)):
                source = sources[i]
                target = targets[i]
                source_landmarks = sources_landmarks[i]
                target_landmarks = targets_landmarks[i]
                source_landmarks_md = sources_landmarks_md[i]
                target_landmarks_md = targets_landmarks_md[i]
                spacing = spacings[i]
                source = source.to(device).view(1, 1, source.size(0), source.size(1), source.size(2))
                target = target.to(device).view(1, 1, target.size(0), target.size(1), target.size(2))
                loss = cost_function(source, target, device=device, **cost_function_params)
                initial_validation_loss += loss.item()
                initial_validation_tre += np.mean(utils.tre(source_landmarks, target_landmarks, spacing))
                initial_validation_md_tre += np.mean(utils.tre(source_landmarks_md, target_landmarks_md, spacing))
    print("Initial training loss: ", initial_training_loss / training_size)
    print("Initial validation loss: ", initial_validation_loss / validation_size)
    print("Initial training TRE: ", initial_training_tre / training_size)
    print("Initial validation TRE: ", initial_validation_tre / validation_size)
    print("Initial training md TRE: ", initial_training_md_tre / training_size)
    print("Initial validation md TRE: ", initial_validation_md_tre / validation_size)

    for epoch in range(num_epochs):
        bet = time.time()
        print("Current epoch: ", str(epoch + 1) + "/" + str(num_epochs))

        # Training
        train_running_loss = 0.0
        train_running_reg = 0.0
        train_running_tre = 0.0
        train_running_tre_md = 0.0
        train_running_ic = 0.0
        model.train()
        for sources, targets, sources_masks, spacings, sources_landmarks, targets_landmarks, sources_landmarks_md, targets_landmarks_md  in training_dataloader:
            for i in range(len(sources)):
                source = sources[i]
                target = targets[i]
                source_landmarks = sources_landmarks[i]
                target_landmarks = targets_landmarks[i]
                source_landmarks_md = sources_landmarks_md[i]
                target_landmarks_md = targets_landmarks_md[i]
                spacing = spacings[i]
                source = source.to(device).view(1, 1, source.size(0), source.size(1), source.size(2))
                target = target.to(device).view(1, 1, target.size(0), target.size(1), target.size(2))

                source = source + tc.rand(source.size()).to(device)*0.000001
                target = target + tc.rand(target.size()).to(device)*0.000001

                with torch.set_grad_enabled(True):
                    df_nonrigid_st = model(source, target)
                    df_nonrigid_ts = model(target, source)
                    composed_df = utc.compose_displacement_fields(df_nonrigid_st, df_nonrigid_ts, device=device)

                    transformed_source = utc.warp_tensor(source, df_nonrigid_st, device=device)
                    transformed_target = utc.warp_tensor(target, df_nonrigid_ts, device=device)

                    cost_st = cost_function(transformed_source, target, device=device, **cost_function_params)
                    cost_ts = cost_function(source, transformed_target, device=device, **cost_function_params)
                    
                    reg_st = alpha * reg_function(df_nonrigid_st, device=device, **reg_function_params)
                    reg_ts = alpha * reg_function(df_nonrigid_ts, device=device, **reg_function_params)

                    ic = tc.mean(composed_df**2)

                    loss = (cost_st + cost_ts) / 2 + (reg_st + reg_ts) / 2 + ic

                    loss.backward()
                    optimizer.step()
                optimizer.zero_grad()

                df_np_nonrigid = utc.tc_df_to_np_df(df_nonrigid_st)

                transformed_target_landmarks = utils.warp_landmarks(target_landmarks, df_np_nonrigid)
                transformed_target_landmarks_md = utils.warp_landmarks(target_landmarks_md, df_np_nonrigid)
                train_running_loss += (cost_st.item() + cost_ts.item()) / 2
                train_running_reg += (reg_st.item() +   reg_ts.item()) / 2
                train_running_tre += np.mean(utils.tre(source_landmarks, transformed_target_landmarks, spacing))
                train_running_tre_md += np.mean(utils.tre(source_landmarks_md, transformed_target_landmarks_md, spacing))
                train_running_ic += ic.item()
            
        print("Train Loss: ", train_running_loss / training_size)
        print("Train Reg: ", train_running_reg / training_size)
        print("Train TRE: ", train_running_tre / training_size)
        print("Train md TRE: ", train_running_tre_md / training_size)
        print("Train IC: ", train_running_ic / training_size)
        train_history.append(train_running_loss / training_size)
        train_reg_history.append(train_running_reg / training_size)
        train_tre_history.append(train_running_tre / training_size)
        train_tre_md_history.append(train_running_tre_md / training_size)
        train_ic_history.append(train_running_ic / training_size)

        # Validation
        val_running_loss = 0.0
        val_running_reg = 0.0
        val_running_tre = 0.0
        val_running_tre_md = 0.0
        val_running_ic = 0.0
        model.eval()
        for sources, targets, sources_masks, spacings, sources_landmarks, targets_landmarks, sources_landmarks_md, targets_landmarks_md  in validation_dataloader:
            for i in range(len(sources)):
                with torch.set_grad_enabled(False):
                    source = sources[i]
                    target = targets[i]
                    source_landmarks = sources_landmarks[i]
                    target_landmarks = targets_landmarks[i]
                    source_landmarks_md = sources_landmarks_md[i]
                    target_landmarks_md = targets_landmarks_md[i]
                    spacing = spacings[i]
                    source = source.to(device).view(1, 1, source.size(0), source.size(1), source.size(2))
                    target = target.to(device).view(1, 1, target.size(0), target.size(1), target.size(2))

                    df_nonrigid_st = model(source, target)
                    df_nonrigid_ts = model(target, source)
                    composed_df = utc.compose_displacement_fields(df_nonrigid_st, df_nonrigid_ts, device=device)

                    transformed_source = utc.warp_tensor(source, df_nonrigid_st, device=device)
                    transformed_target = utc.warp_tensor(target, df_nonrigid_ts, device=device)

                    cost_st = cost_function(transformed_source, target, device=device, **cost_function_params)
                    cost_ts = cost_function(source, transformed_target, device=device, **cost_function_params)
                    
                    reg_st = alpha * reg_function(df_nonrigid_st, device=device, **reg_function_params)
                    reg_ts = alpha * reg_function(df_nonrigid_ts, device=device, **reg_function_params)

                    ic = tc.mean(composed_df**2)

                    df_np_nonrigid = utc.tc_df_to_np_df(df_nonrigid_st)

                    transformed_target_landmarks = utils.warp_landmarks(target_landmarks, df_np_nonrigid)
                    transformed_target_landmarks_md = utils.warp_landmarks(target_landmarks_md, df_np_nonrigid)
                    val_running_loss += (cost_st.item() + cost_ts.item()) / 2
                    val_running_reg += (reg_st.item() +   reg_ts.item()) / 2
                    val_running_tre += np.mean(utils.tre(source_landmarks, transformed_target_landmarks, spacing))
                    val_running_tre_md += np.mean(utils.tre(source_landmarks_md, transformed_target_landmarks_md, spacing))
                    val_running_ic += ic.item()

        print("Val Loss: ", val_running_loss / validation_size)
        print("Val Reg: ", val_running_reg / validation_size)
        print("Val TRE: ", val_running_tre / validation_size)
        print("Val md TRE: ", val_running_tre_md / validation_size)
        print("Val IC: ", val_running_ic / validation_size)
        val_history.append(val_running_loss / validation_size)
        val_reg_history.append(val_running_reg / validation_size)
        val_tre_history.append(val_running_tre / validation_size)
        val_tre_md_history.append(val_running_tre_md / validation_size)
        val_ic_history.append(val_running_ic / validation_size)

        scheduler.step()

        eet = time.time()
        print("Epoch time: ", eet - bet, "seconds.")
        print("Estimated time to end: ", (eet - bet)*(num_epochs-epoch), "seconds.")

    if model_save_path is not None:
        tc.save(model.state_dict(), model_save_path)

    plt.figure()
    plt.plot(train_history, "r-")
    plt.plot(val_history, "b-")
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Train", "Validation"])
    plt.savefig(os.path.join(figures_path, model_name + "_cost.png"), bbox_inches = 'tight', pad_inches = 0)

    plt.figure()
    plt.plot(train_reg_history, "r-")
    plt.plot(val_reg_history, "b-")
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("Reg")
    plt.legend(["Train", "Validation"])
    plt.savefig(os.path.join(figures_path, model_name + "_reg.png"), bbox_inches = 'tight', pad_inches = 0)

    plt.figure()
    plt.plot(train_ic_history, "r-")
    plt.plot(val_ic_history, "b-")
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("IC")
    plt.legend(["Train", "Validation"])
    plt.savefig(os.path.join(figures_path, model_name + "_ic.png"), bbox_inches = 'tight', pad_inches = 0)

    plt.figure()
    plt.plot(train_tre_history, "r-")
    plt.plot(val_tre_history, "b-")
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("TRE [mm]")
    plt.legend(["Train", "Validation"])
    plt.savefig(os.path.join(figures_path, model_name + "_tre.png"), bbox_inches = 'tight', pad_inches = 0)

    plt.figure()
    plt.plot(train_tre_md_history, "r-")
    plt.plot(val_tre_md_history, "b-")
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("TRE MD [mm]")
    plt.legend(["Train", "Validation"])
    plt.savefig(os.path.join(figures_path, model_name + "_tre_md.png"), bbox_inches = 'tight', pad_inches = 0)

    plt.show()


def testing(testing_params):
    model_name = testing_params['model_name']
    model_save_path = os.path.join(models_path, model_name)

    model = nn.load_network(device, path=model_save_path)
    model = model.to(device)

    cost_function = cf.ncc_local_tc
    cost_function_params = dict()

    reg_function = rg.diffusion_tc

    batch_size = 1
    ids = range(1, 16)
    loader = dl.UnsupervisedLoader(data_path, ids, transforms=None)
    dataloader = torch.utils.data.DataLoader(loader, batch_size = batch_size, shuffle = False, num_workers = 4, collate_fn = dl.collate_to_list_unsupervised)

    model.eval()
    for sources, targets, sources_masks, spacings, sources_landmarks, targets_landmarks, sources_landmarks_md, targets_landmarks_md  in dataloader:
        with torch.set_grad_enabled(False):
            for i in range(len(sources)):
                source = sources[i]
                target = targets[i]
                source_landmarks = sources_landmarks[i]
                target_landmarks = targets_landmarks[i]
                source_landmarks_md = sources_landmarks_md[i]
                target_landmarks_md = targets_landmarks_md[i]
                spacing = spacings[i]
                source = source.to(device).view(1, 1, source.size(0), source.size(1), source.size(2))
                target = target.to(device).view(1, 1, target.size(0), target.size(1), target.size(2))

                print("Shape: ", source.size())

                df_nonrigid = model(source, target)

                transformed_source = utc.warp_tensor(source, df_nonrigid, device=device)
                df_np_nonrigid = utc.tc_df_to_np_df(df_nonrigid)

                transformed_target_landmarks = utils.warp_landmarks(target_landmarks, df_np_nonrigid)
                transformed_target_landmarks_md = utils.warp_landmarks(target_landmarks_md, df_np_nonrigid)

                loss = cost_function(transformed_source, target, device=device, **cost_function_params)

                print("Initial Loss: ", cost_function(source, target, device=device, **cost_function_params).item())
                print("Final Loss: ", loss.item())
                print("Reg: ", reg_function(df_nonrigid, device=device))
                print("Initial TRE: ", np.mean(utils.tre(source_landmarks, target_landmarks, spacing)))
                print("Registered TRE: ", np.mean(utils.tre(source_landmarks, transformed_target_landmarks, spacing)))
                print("Initial MD TRE: ", np.mean(utils.tre(source_landmarks_md, target_landmarks_md, spacing)))
                print("Registered MD TRE: ", np.mean(utils.tre(source_landmarks_md, transformed_target_landmarks_md, spacing)))

                c_slice = 40
                plt.figure()
                plt.subplot(1, 3, 1)
                plt.imshow(source[0, 0, :, :, :].detach().cpu().numpy()[:, :, c_slice], cmap='gray')
                plt.axis('off')
                plt.subplot(1, 3, 2)
                plt.imshow(target[0, 0, :, :, :].detach().cpu().numpy()[:, :, c_slice], cmap='gray')
                plt.axis('off')
                plt.subplot(1, 3, 3)
                plt.imshow(transformed_source[0, 0, :, :, :].detach().cpu().numpy()[:, :, c_slice], cmap='gray')
                plt.axis('off')

                plt.figure()
                plt.subplot(1, 3, 1)
                plt.imshow(df_np_nonrigid[0, :, :, c_slice], cmap='gray')
                plt.subplot(1, 3, 2)
                plt.imshow(df_np_nonrigid[1, :, :, c_slice], cmap='gray')
                plt.subplot(1, 3, 3)
                plt.imshow(df_np_nonrigid[2, :, :, c_slice], cmap='gray')
                plt.show()


def run():
    training_params = dict()
    training_params['model_name'] = None # TO DO
    training_params['num_epochs'] = None # TO DO
    training_params['batch_size'] = None # TO DO
    training_params['learning_rate'] = None # TO DO
    training_params['initial_path'] = None
    training_params['decay_rate'] = None # TO DO
    training_params['alpha'] = None # TO DO
    training(training_params)

    testing_params = dict()
    testing_params['model_name'] = None # TO DO
    testing(testing_params)





if __name__ == "__main__":
    run()