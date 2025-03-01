import numpy as np
import os
gpuIdxStr = '1'

random_seed = 2023+int(gpuIdxStr)

exp_name = 'HypothesisGeneralizaton'

split_based_on, num_x = 'hypothesis', 5

prefix = f'python new_exp.py --gpu {gpuIdxStr} --random_seed {random_seed} --exp_name {exp_name} --split_based_on {split_based_on} --num_x {num_x} --wandb 1 --epochs 512'

lr_list = [0.00002]#, 0.00005, 0.00010]

wd_list = [0.0005] #[0.0002, 0.0005]

batch_size_list = [16]#, 32, 64]

modelName_list = ['dual']#, 'nano']

loss_on_list = ['all'] #['all', 'y\&z']

icl_sampling_list = ['iid']

h_prefix_format_list = [0]

for lr in lr_list:
    for wd in wd_list:
        for batch_size in batch_size_list:
            for modelName in modelName_list:
                for loss_on in loss_on_list:
                    for icl_sampling in icl_sampling_list:
                        for h_prefix_format in h_prefix_format_list:
                            hyper = f'--lr {lr} --wd {wd} --batch_size {batch_size} --modelName {modelName} --loss_on {loss_on} --icl_sampling {icl_sampling} --h_prefix_format {h_prefix_format}'
                            cl = f'{prefix} {hyper}'
                            #print(prefix)
                            #print(hyper)
                            #print(cl)
                            os.system(cl)