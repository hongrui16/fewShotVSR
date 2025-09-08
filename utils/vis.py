# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import os
import cv2
import numpy as np
import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# from config import cfg

os.environ["PYOPENGL_PLATFORM"] = "egl"

import matplotlib.pyplot as plt
import os, sys


def visualize_loss(loss_epoch_list, split, save_path, 
                   loss_epoch_list_2 = [], split_2 = None, 
                   ele_list_3 = [], ele_name_3 = None,
                   epoch = 0, logger = None
                   ):
    
    
    # print('loss_epoch_list', loss_epoch_list)
    # print('loss_epoch_list_2', loss_epoch_list_2)
    # Extract unique loss keys from dictionaries
    if len(loss_epoch_list) == 0:
        return
    epoch_loss_dict = loss_epoch_list[0]
    loss_keys = list(epoch_loss_dict.keys())

    if len(loss_epoch_list_2) > 0:
        loss_keys_2 = list(loss_epoch_list_2[0].keys())
    else:
        loss_keys_2 = []

    union_keys = set(loss_keys + loss_keys_2)
    
    # sort union_keys
    union_keys = sorted(list(union_keys))

    # print('union_keys', union_keys)

    # Number of subplots needed
    num_figs = len(union_keys)
    if len(ele_list_3) > 0:
        num_figs += len(ele_list_3[0].keys())

    
    num_rows = 1 + (num_figs // 4)
    num_cols = min(num_figs, 4)
    # print('num_rows', num_rows, 'num_cols', num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    if num_figs == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, key in enumerate(union_keys):
        values = []
        for epoch_dict in loss_epoch_list:
            if not key in epoch_dict:
                continue
            values.append(epoch_dict.get(key, 0))
        axes[i].plot(range(epoch, epoch + len(values)), values, label=split, marker='o', color='b')
        if len(loss_epoch_list_2) > 0:
            values_2 = []
            for epoch_dict in loss_epoch_list_2:
                if not key in epoch_dict:
                    continue
                values_2.append(epoch_dict.get(key, 0))
            axes[i].plot(range(epoch, epoch + len(values_2)), values_2, label=split_2, marker='x', color='r')
        axes[i].set_title(f'{key}')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Value')
        axes[i].legend()
        axes[i].grid(True)

    if len(ele_list_3) > 0:
        ele_dict = ele_list_3[0]
        ele_keys = list(ele_dict.keys())
        for j, key in enumerate(ele_keys):
            values = [epoch_dict.get(key, 0) for epoch_dict in ele_list_3]
            axes[i + j + 1].plot(range(epoch, epoch + len(values)), values, label=ele_name_3, marker='x', color='r')
            axes[i + j + 1].set_title(f'{key}')
            axes[i + j + 1].set_xlabel('Epoch')
            axes[i + j + 1].set_ylabel(f'{ele_name_3}')
            axes[i + j + 1].legend()
            axes[i + j + 1].grid(True)

    # Hide unused subplots if any
    for i in range(num_figs, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    # plt.show()

    if os.path.exists(save_path):
        os.remove(save_path)
    plt.savefig(save_path)
    if logger is not None:
        logger.info(f"epoch: {epoch}, Saved visualization to {save_path}")
    plt.close()




if __name__ == '__main__':
    train_loss_list = []
    val_loss_list = []
    for i in range(10):
        train_loss_list.append({'loss': i, 'loss2': i*2})
        val_loss_list.append({'loss': i+1, 'loss2': (i+1)*2})
    # visualize_loss(train_loss_list, 'train', 'train.png', val_loss_list, 'val')

    loss_epoch_list = [{'x0_loss': 0.1280915311404637}]
    loss_epoch_list_2 = [{'x0_loss': 0.13798222371510097}]
    visualize_loss(loss_epoch_list, 'train', 'train.png', loss_epoch_list_2, 'val')