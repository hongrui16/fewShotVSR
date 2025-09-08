# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import os
import os.path as osp
import sys
import numpy as np
import argparse

class Config:
    
        
    train_batch_size = 20

    ### dataset config
    dataset_name = 'DummyDataset' # InterHand26M, InterHand26M_SHMulMod
    
    
    test_batch_size = train_batch_size # 50 for 80g gpu
    shuffle_train_set = True
    process_single_hand_img = True

    
    ### training config
    num_train_epochs = 60
    dump_samples = False
    warmup_ratio = 0.1


    use_latent_warp = True
    use_adaptive_gate = True

    network_name = 'fewShotVSR' # HandPoseNetV2, HandPoseNet


    # training Config        
    resume_path = None # path to the resume weight, if None, do not resume training.
    finetune = True
    
    
    
    ## optimizer config
    
    optimizer_dict = {
        'name': 'adam', # 'adam' or 'SGD'
        'lr': 5*1e-4,
        'weight_decay': 1e-2,

        ## adam config
        'adam_betas': (0.9, 0.999),
        'use_8bit_adam': False,
        'adam_epsilon': 1e-8,
    }

    ### output config
    weight_output_dir = '/scratch/rhong5/weights/temp_training_weights'
    log_dir = 'zlog'

cfg = Config()



def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")

    parser.add_argument("--debug", action="store_true", help="Run in debug mode.")

    parser.add_argument("--mode", default='train', help="runtime mode.")
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps.")

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )

    

    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    
    
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default='fp16',
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_false", help="Whether or not to use xformers."
    )
    
    
    


    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()




    return args
