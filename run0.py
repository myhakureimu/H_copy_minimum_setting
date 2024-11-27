import numpy as np
import os
gpuIdxStr = '0'

random_seed = 2023+int(gpuIdxStr)

prefix = f'python new_exp.py --gpu {gpuIdxStr} --random_seed {random_seed} --wandb 1 --epochs 128'

lr_list = [0.00005]

wd_list = [0.0005, 0.0001, 0.0002]

batch_size_list = [128]

modelName_list = ['dual']#, 'nano']

loss_on_list = ['all', 'icl\&\>z', 'y\&z']#, 'z']

for lr in lr_list:
    for wd in wd_list:
        for batch_size in batch_size_list:
            for modelName in modelName_list:
                for loss_on in loss_on_list:
                    hyper = f'--lr {lr} --wd {wd} --batch_size {batch_size} --modelName {modelName} --loss_on {loss_on}'
                    cl = f'{prefix} {hyper}'
                    #print(prefix)
                    #print(hyper)
                    #print(cl)
                    os.system(cl)