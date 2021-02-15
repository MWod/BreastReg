import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import random
import SimpleITK as sitk


import torch as tc
import torch.utils as tcu

import utils as u


def collate_to_list_unsupervised(batch):
    sources = [item[0].view(item[0].size(0), item[0].size(1), item[0].size(2)) for item in batch]
    targets = [item[1].view(item[1].size(0), item[1].size(1), item[1].size(2)) for item in batch]
    sources_masks = [item[2].view(item[2].size(0), item[2].size(1), item[2].size(2)) for item in batch]
    spacing = [item[3] for item in batch]
    sources_landmarks = [item[4] for item in batch]
    targets_landmarks = [item[5] for item in batch]
    sources_landmarks_md = [item[6] for item in batch]
    targets_landmarks_md = [item[7] for item in batch]
    return sources, targets, sources_masks, spacing, sources_landmarks, targets_landmarks, sources_landmarks_md, targets_landmarks_md

class UnsupervisedLoader(tcu.data.Dataset):
    def __init__(self, data_path, case_ids, transforms=None):
        self.data_path = data_path
        self.all_ids = case_ids
        self.transforms = transforms

    def __len__(self):
        return len(self.all_ids)

    def __getitem__(self, idx):
        case_id = self.all_ids[idx]
        source, target, source_mask, spacing, source_landmarks, target_landmarks, source_landmarks_md, target_landmarks_md = u.load_case(self.data_path, case_id)
        source = u.normalize(source)
        target = u.normalize(target)

        if self.transforms:
            source = self.transforms(source)
            target = self.transforms(target)
            source_mask = self.transforms(source_mask)
            source_landmarks = self.transforms(source_landmarks)
            target_landmarks = self.transforms(target_landmarks)
            source_landmarks_md = self.transforms(source_landmarks_md)
            target_landmarks_md = self.transforms(target_landmarks_md)

        source_tensor, target_tensor = tc.from_numpy(source.astype(np.float32)), tc.from_numpy(target.astype(np.float32))
        source_mask_tensor  = tc.from_numpy(source_mask.astype(np.float32))
        return source_tensor, target_tensor, source_mask_tensor, spacing, source_landmarks, target_landmarks, source_landmarks_md, target_landmarks_md
