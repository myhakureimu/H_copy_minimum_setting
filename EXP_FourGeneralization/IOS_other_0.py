import numpy as np
import os
gpuIdxStr = '0'

random_seed = 2023+int(gpuIdxStr)

training_content = 'h+xy+z'

HEAD, exp_name = 'FourGeneralization', 'IOHypothesis+Size'

num_x, num_y = 5, 2
icl_k, max_table_length = 5, 16
num_training_hypotheses, num_training_tables = 0, 0

epochs = 768

prefix = f'python icl_exp.py --gpu {gpuIdxStr} --random_seed {random_seed} --wandb 1 --epochs {epochs} \
        --HEAD {HEAD} --training_content {training_content} --exp_name {exp_name} \
        --icl_k {icl_k} --num_x {num_x} --num_y {num_y} \
        --num_training_hypotheses {num_training_hypotheses} \
        --max_table_length {max_table_length} --num_training_tables {num_training_tables}'

use_scheduler = 1
depth_list = [2, 2, 2]#, 8]
lr_list = [0.001, 0.001, 0.0005]#, 0.00002]
modelName_list = ['lstm', 'gru', 'mamba']#, 'dual'] #, 'nano']

wd_list = [0.0005]
batch_size_list = [16] #, 32, 64]
loss_on_list = ['all'] #['all', 'y\&z']
icl_sampling_list = ['iid']
h_prefix_format_list = [0]
icl_y_noise_list = [0.0]
for depth, lr, modelName in zip(depth_list, lr_list, modelName_list):
    for wd in wd_list:
        for batch_size in batch_size_list:
            for loss_on in loss_on_list:
                for icl_sampling in icl_sampling_list:
                    for h_prefix_format in h_prefix_format_list:
                        for icl_y_noise in icl_y_noise_list:
                            hyper = f'--depth {depth} --lr {lr} --use_scheduler {use_scheduler} --wd {wd} --batch_size {batch_size} --modelName {modelName}\
                                    --loss_on {loss_on} --icl_sampling {icl_sampling} --h_prefix_format {h_prefix_format} --icl_y_noise {icl_y_noise}'
                            cl = f'{prefix} {hyper}'
                            #print(prefix)
                            #print(hyper)
                            #print(cl)
                            os.system(cl)