import numpy as np
import os
gpuIdxStr = '0'

random_seed = 2023+int(gpuIdxStr)

HEAD = 'NUMTRAIN'

exp_name = 'TableGeneralization'

split_based_on, icl_k, num_x, num_y = 'table', 4, 4, 2

max_table_length = 4

num_training_tables = 2**6

epochs = 768

prefix = f'python new_exp.py --gpu {gpuIdxStr} --random_seed {random_seed} --wandb 1 --epochs {epochs} \
        --HEAD {HEAD} --exp_name {exp_name} --split_based_on {split_based_on} \
        --num_x {num_x} --num_y {num_y} \
        --max_table_length {max_table_length} --num_training_tables {num_training_tables}'

depth_list = [8]

lr_list = [0.00002]

wd_list = [0.0005]

batch_size_list = [16] #, 32, 64]

modelName_list = ['dual'] #, 'nano']

loss_on_list = ['all'] #['all', 'y\&z']

icl_sampling_list = ['iid']

h_prefix_format_list = [0]

icl_y_noise_list = [0.0]

for depth in depth_list:
    for lr in lr_list:
        for wd in wd_list:
            for batch_size in batch_size_list:
                for modelName in modelName_list:
                    for loss_on in loss_on_list:
                        for icl_sampling in icl_sampling_list:
                            for h_prefix_format in h_prefix_format_list:
                                for icl_y_noise in icl_y_noise_list:
                                    hyper = f'--depth {depth} --lr {lr} --wd {wd} --batch_size {batch_size} --modelName {modelName}\
                                          --loss_on {loss_on} --icl_sampling {icl_sampling} --h_prefix_format {h_prefix_format} --icl_y_noise {icl_y_noise}'
                                    cl = f'{prefix} {hyper}'
                                    #print(prefix)
                                    #print(hyper)
                                    #print(cl)
                                    os.system(cl)