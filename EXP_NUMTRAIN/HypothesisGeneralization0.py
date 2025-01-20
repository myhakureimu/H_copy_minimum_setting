import numpy as np
import os
gpuIdxStr = '0'

shift_list = [2023, 2024, 2025, 2026]
for shift in shift_list:
    random_seed = shift + int(gpuIdxStr)

    HEAD = 'NUMTRAIN-v1.1'

    exp_name = 'HypothesisGeneralization'

    split_based_on, icl_k, num_x, num_y = 'hypothesis', 5, 5, 2

    max_table_length = 4

    num_training_tables = 2**0

    prefix = f'python new_exp.py --gpu {gpuIdxStr} --random_seed {random_seed} --wandb 1 \
            --HEAD {HEAD} --exp_name {exp_name} --split_based_on {split_based_on} \
            --num_x {num_x} --num_y {num_y} \
            --max_table_length {max_table_length} --num_training_tables {num_training_tables}'

    epochs_list = [256, 1024]
    depth_list = [8, 8]

    lr_list = [0.0005, 0.00002]

    wd_list = [0.0005]

    batch_size_list = [16] #, 32, 64]

    modelName_list = ['dual', 'dual'] #, 'nano']

    loss_on_list = ['all'] #['all', 'y\&z']

    icl_sampling_list = ['iid']

    h_prefix_format_list = [0]

    icl_y_noise_list = [0.0]

    for epochs, depth, lr, modelName in zip(epochs_list, depth_list, lr_list, modelName_list):
        for wd in wd_list:
            for batch_size in batch_size_list:
                for loss_on in loss_on_list:
                    for icl_sampling in icl_sampling_list:
                        for h_prefix_format in h_prefix_format_list:
                            for icl_y_noise in icl_y_noise_list:
                                hyper = f'--epochs {epochs} --depth {depth} --lr {lr} --wd {wd} --batch_size {batch_size} --modelName {modelName}\
                                    --loss_on {loss_on} --icl_sampling {icl_sampling} --h_prefix_format {h_prefix_format} --icl_y_noise {icl_y_noise}'
                                cl = f'{prefix} {hyper}'
                                #print(prefix)
                                #print(hyper)
                                #print(cl)
                                os.system(cl)