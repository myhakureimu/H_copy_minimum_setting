import os
import argparse
import setproctitle
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import time
import random
import numpy as np
import wandb

from new_hmanager import HypothesisManager, DataloaderManager
from find_valid_zs import find_valid_zs

matplotlib.rc('text', usetex=True)
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')
# python train.py --name=GPT

# environment setting
parser = argparse.ArgumentParser(description='PyTorch In-context Learning Training Code')
parser.add_argument('--gpu', default='0', type=str, help='which gpus to use')
parser.add_argument('--random_seed', default=1, type=int, help='the seed used for torch & numpy')
parser.add_argument('--wandb', default=0, type=int)

parser.add_argument('--HEAD', default='TEST', type=str)
parser.add_argument('--exp_name', default='TableLengthGeneralization', type=str)
#arxived args
# parser.add_argument('--SigmaRe', default=2, type=int)
# parser.add_argument('--NormAtt', default=0, type=int)
# parser.add_argument('--FirstLayerNorm', default=1, type=int)


# H setting for init hypothesismanager
''' parser.add_argument('--mode', default='binary', type=str, choices=['binary', 'permutation'])  #binary only '''
parser.add_argument('--num_x', default=4, type=int)
parser.add_argument('--num_y', default=2, type=int)
parser.add_argument('--num_training_tables', default=0, type=int)
parser.add_argument('--max_table_length', default=8, type=int)
# table_lengths
parser.add_argument('--split_based_on', default='table', type=str)
# split_ratio
# train_info
# test__info

# H+ICL format for dataloadermanager
parser.add_argument('--icl_k', default=4, type=int)
parser.add_argument('--loss_on', default='all', type=str, choices=['all', 'icl&>z', 'y&z', 'z'], help = 'all=prefix&icl&z, icl=x&y&>')
parser.add_argument('--icl_sampling', default='iid', type=str, choices = ['ordered', 'shuffle', 'iid', 'optimal', 'mix'])
parser.add_argument('--sampling_disparity', default=1.0, type=float)
parser.add_argument('--icl_y_noise', default=0.0, type=float)
parser.add_argument('--h_prefix_format', default=0, type=int, choices=[0,1])
parser.add_argument('--mix_prob_train1', default=0.5, type=float)

# model setting
parser.add_argument('--modelName', default='dual', type=str, choices=['dual', 'mamba', 'lstm', 'gru'])
parser.add_argument('--num_heads', default=2, type=int, help='number of heads for multi-headed attention (default: 8)')
parser.add_argument('--depth', default=2, type=int, help='depth of the transformer architecture (default: 12)')
parser.add_argument('--embed_dim', default=128, type=int, help='embedding dimension of the transformer feature extractor (default: 256)')
# parser.add_argument('--dropout', default=0.0, type=float, help='dropout')
parser.add_argument('--llm_max_length', default=256, type=int, help='maximum sequence length of the input (default: 11)')

#optimization
parser.add_argument('--lr', default=0.00002, type=float, help='initial model learning rate') #0.0005
parser.add_argument('--wd', default=0.0005, type=float, help='weight decay hyperparameter (default: 0.00001)') #0.1
parser.add_argument('--batch_size', default=16, type=int, help='mini-batch size (default: 64)') #32
parser.add_argument('--n_steps', default=512, type=int, help='total number of training steps we want to run')
parser.add_argument('--epochs', default=512, type=int, help='number of total epochs to run')

parser.set_defaults(augment=True)
args = parser.parse_args()

args.n_steps = int(1024 * 16 / args.batch_size)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
setproctitle.setproctitle(f'{args.exp_name} {args.sampling_disparity} {args.random_seed}')

import torch
import torch.nn as nn
if args.modelName == 'nano':
    from utils.nano_gpt import GPTConfig, NanoGPT
if args.modelName == 'pytorch':
    from utils.pytorch_transformer import PytorchTransformer
if args.modelName == 'dual':
    from utils.models import TransformerModel
if args.modelName == 'mamba':
    from utils.ssm import KLayerMambaModel
if args.modelName == 'lstm':
    from utils.lstm import LSTMModel
if args.modelName == 'gru':
    from utils.gru import GRUModel

torch.backends.cudnn.benchmark = True
#torch.manual_seed(args.random_seed)
#np.random.seed(args.random_seed)
#random.seed(args.random_seed)

def generate_normalized_vector(n, sampling_disparity):
    if n < 1:
        raise ValueError("n must be a positive integer.")

    half = n // 2
    if n % 2 == 0:  # Even case
        vector = [1] * half + [sampling_disparity] * half
    else:  # Odd case
        vector = [1] * half + [sampling_disparity**0.5] + [sampling_disparity] * half

    # Normalize the vector
    vector = np.array(vector, dtype=float)
    normalized_vector = vector / vector.sum()

    return normalized_vector

class AverageMeter(object):
    def __init__(self, table_lengths):
        self.table_lengths = table_lengths
        self.reset()

    def reset(self):
        self.val = {0:0}
        self.avg = {0:0}
        self.sum = {0:0}
        self.count = {0:0}
        for l in self.table_lengths:
            self.val[l] = 0
            self.avg[l] = 0
            self.sum[l] = 0
            self.count[l] = 0

    def update(self, table_lengths, vals, counts):
        #print('vals.shape =', vals.shape)
        for i in range(len(table_lengths)):
            table_length = table_lengths[i]
            val = vals[i]
            count = counts[i]
            self.val[table_length] = val
            self.sum[0] += val
            self.sum[table_length] += val
            self.count[0] += count
            self.count[table_length] += count
            self.avg[0] = self.sum[0] / (0.000001+self.count[0])
            self.avg[table_length] = self.sum[table_length] / (0.000001+self.count[table_length])

def pad4input(xy_batch, mask_batch, h_prefix, h_prefix_mask, pad_token):
    hxy_list = []
    mask_list = []
    for i in range(len(xy_batch)):
        hxy_i = torch.cat([h_prefix[i], xy_batch[i]], dim=0)
        mask_i = torch.cat([h_prefix_mask[i], mask_batch[i]], dim=0)
        hxy_list.append(hxy_i)
        mask_list.append(mask_i)

    # Step 3: Pad sequences to the maximum length in the batch
    # Determine the maximum sequence length
    max_seq_len = max([seq.size(0) for seq in hxy_list])

    # Pad hxy_list and mask_list
    hxy_padded = []
    mask_padded = []
    for hxy_seq, mask_seq in zip(hxy_list, mask_list):
        seq_len = hxy_seq.size(0)
        pad_len = max_seq_len - seq_len
        # Pad hxy_seq with pad_token
        hxy_seq_padded = torch.cat([hxy_seq, torch.full((pad_len,), pad_token, dtype=hxy_seq.dtype, device=hxy_seq.device)])
        # Pad mask_seq with 0
        mask_seq_padded = torch.cat([mask_seq, torch.zeros(pad_len, dtype=mask_seq.dtype, device=mask_seq.device)])
        hxy_padded.append(hxy_seq_padded)
        mask_padded.append(mask_seq_padded)

    # Stack the padded sequences to create tensors
    hxy = torch.stack(hxy_padded, dim=0)
    mask = torch.stack(mask_padded, dim=0)

    return hxy, mask

def l2s(float_list):
    if float_list is None:
        return '[None]'
    # Format each float to have exactly three decimal places as a string
    formatted_probs = [f"{prob:.3f}" for prob in float_list]
    
    # Combine the formatted strings into a list string
    output_str = "[" + ", ".join(formatted_probs) + "]"
    
    return output_str
# strings = {}
# strings['train'] = {}
# strings['test_'] = {}
def train_model(args, phase, table_lengths, dmanager, model, optimizer, epoch):
    wandb_info = {}
    if phase not in ['train', 'test_']:
        raise Exception(f'phase = {phase} is not valid')
    icl_sampling = dmanager.icl_sampling
    iid_probability = dmanager.iid_probability
    dataloader = dmanager.get_pytorch_dataloader()
    int2token = dmanager.hmanager.int2token
    tokens = dmanager.tokens
    #print(int2token)

    loss_f = torch.nn.CrossEntropyLoss(reduction = 'none')
    
    batch_loss = AverageMeter(table_lengths)
    batch_acc_x = AverageMeter(table_lengths)
    batch_acc_y = AverageMeter(table_lengths)
    batch_acc_z = AverageMeter(table_lengths)
    batch_acc_h = AverageMeter(table_lengths)
    batch_acc_s = AverageMeter(table_lengths)
    batch_acc_zs = AverageMeter(table_lengths)
    if phase == 'train':
        model.train()
    if phase == 'test_':
        model.eval()

    if phase in ['test1', 'test2', 'test4', 'test8']:
        batch_acc_ = AverageMeter(table_lengths)
        batch_acch = AverageMeter(table_lengths)
        model.eval()
    
    context = torch.enable_grad() if phase == 'train' else torch.no_grad()
    with context:
        for batch in (pbar := tqdm(dataloader)):
            table_length_batch = [H.shape[0] for H in batch['H_list']]

            spH_prefix       = torch.stack(batch['spH_prefix_list_info']['spH_prefix_list'      ]).cuda().to(torch.long)
            spH_prefix_xmask = torch.stack(batch['spH_prefix_list_info']['spH_prefix_xmask_list']).cuda().to(torch.float32)
            spH_prefix_ymask = torch.stack(batch['spH_prefix_list_info']['spH_prefix_ymask_list']).cuda().to(torch.float32)
            spH_prefix_zmask = torch.stack(batch['spH_prefix_list_info']['spH_prefix_zmask_list']).cuda().to(torch.float32)
            spH_prefix_hmask = torch.stack(batch['spH_prefix_list_info']['spH_prefix_hmask_list']).cuda().to(torch.float32)
            spH_prefix_smask = torch.stack(batch['spH_prefix_list_info']['spH_prefix_smask_list']).cuda().to(torch.float32)
            
            xy_seq       = torch.stack(batch['xy_seq_list_info']['xy_seq_list'      ]).cuda().to(torch.long)
            xy_seq_xmask = torch.stack(batch['xy_seq_list_info']['xy_seq_xmask_list']).cuda().to(torch.float32)
            xy_seq_ymask = torch.stack(batch['xy_seq_list_info']['xy_seq_ymask_list']).cuda().to(torch.float32)
            xy_seq_zmask = torch.stack(batch['xy_seq_list_info']['xy_seq_zmask_list']).cuda().to(torch.float32)
            xy_seq_hmask = torch.stack(batch['xy_seq_list_info']['xy_seq_hmask_list']).cuda().to(torch.float32)
            xy_seq_smask = torch.stack(batch['xy_seq_list_info']['xy_seq_smask_list']).cuda().to(torch.float32)

            z_suffix       = torch.stack(batch['z_suffix_list_info']['z_suffix_list'      ]).cuda().to(torch.long)
            z_suffix_xmask = torch.stack(batch['z_suffix_list_info']['z_suffix_xmask_list']).cuda().to(torch.float32)
            z_suffix_ymask = torch.stack(batch['z_suffix_list_info']['z_suffix_ymask_list']).cuda().to(torch.float32)
            z_suffix_zmask = torch.stack(batch['z_suffix_list_info']['z_suffix_zmask_list']).cuda().to(torch.float32)
            z_suffix_hmask = torch.stack(batch['z_suffix_list_info']['z_suffix_hmask_list']).cuda().to(torch.float32)
            z_suffix_smask = torch.stack(batch['z_suffix_list_info']['z_suffix_smask_list']).cuda().to(torch.float32)

            # print('*'*20)
            # print(spH_prefix[0])
            # print(xy_seq[0])
            # print(z_suffix[0])
            # break

            all_seq = torch.cat([spH_prefix, xy_seq, z_suffix], dim=1)

            i_seq = all_seq[:, :-1]
            o_seq = all_seq[:, 1:]

            p_seq = model.forward(i_seq)
            
            losses = loss_f(p_seq.transpose(1, 2), o_seq.to(torch.long))

            if args.loss_on == 'all':
                #loss = torch.mean(losses)
                losses_mask = torch.cat([torch.ones_like(spH_prefix),
                                        torch.ones_like(xy_seq),
                                        torch.ones_like(z_suffix)], dim=1).float()
            elif args.loss_on == 'icl&>z':
                losses_mask = torch.cat([torch.zeros_like(spH_prefix),
                                        torch.ones_like(xy_seq),
                                        torch.ones_like(z_suffix)], dim=1).float()
            elif args.loss_on == 'y&z':
                losses_mask = torch.cat([torch.zeros_like(spH_prefix),
                                        xy_seq_ymask,
                                        z_suffix_zmask], dim=1).float()
            elif args.loss_on == 'z':
                losses_mask = torch.cat([torch.zeros_like(spH_prefix),
                                        torch.zeros_like(xy_seq),
                                        z_suffix_zmask], dim=1).float()
            losses_mask = losses_mask[:, 1:]
            loss = torch.sum(losses*losses_mask)/torch.sum(losses_mask)

            if phase == 'train':
                # Backpropagate gradients and update model
                optimizer.zero_grad()
                model.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                loss_per_h  = torch.sum(losses*losses_mask, dim=1)
                count_per_h = torch.sum(losses_mask, dim=1)

                correct = (torch.argmax(p_seq, dim=2) == o_seq)

                correct_xmask = torch.cat([spH_prefix_xmask, xy_seq_xmask, z_suffix_xmask], dim=1).to(torch.float32)[:, 1:]
                correct_ymask = torch.cat([spH_prefix_ymask, xy_seq_ymask, z_suffix_ymask], dim=1).to(torch.float32)[:, 1:]
                correct_zmask = torch.cat([spH_prefix_zmask, xy_seq_zmask, z_suffix_zmask], dim=1).to(torch.float32)[:, 1:]
                correct_hmask = torch.cat([spH_prefix_hmask, xy_seq_hmask, z_suffix_hmask], dim=1).to(torch.float32)[:, 1:]
                correct_smask = torch.cat([spH_prefix_smask, xy_seq_smask, z_suffix_smask], dim=1).to(torch.float32)[:, 1:]

                correct_x_per_h = torch.sum(correct*correct_xmask, dim=1)
                correct_y_per_h = torch.sum(correct*correct_ymask, dim=1)
                correct_z_per_h = torch.sum(correct*correct_zmask, dim=1)
                correct_h_per_h = torch.sum(correct*correct_hmask, dim=1)
                correct_s_per_h = torch.sum(correct*correct_smask, dim=1)

                count_x_per_h = torch.sum(correct_xmask, dim=1)
                count_y_per_h = torch.sum(correct_ymask, dim=1)
                count_z_per_h = torch.sum(correct_zmask, dim=1)
                count_h_per_h = torch.sum(correct_hmask, dim=1)
                count_s_per_h = torch.sum(correct_smask, dim=1)

            # Record the loss and elapsed time
            batch_loss .update(table_length_batch, loss_per_h     .data, count_per_h)
            batch_acc_x.update(table_length_batch, correct_x_per_h.data, count_x_per_h)
            batch_acc_y.update(table_length_batch, correct_y_per_h.data, count_y_per_h)
            batch_acc_z.update(table_length_batch, correct_z_per_h.data, count_z_per_h)
            batch_acc_h.update(table_length_batch, correct_h_per_h.data, count_h_per_h)
            batch_acc_s.update(table_length_batch, correct_s_per_h.data, count_s_per_h)
            
            if phase == 'test_':
                p_seq = torch.argmax(p_seq, dim=2)[correct_zmask==1]
                ##print(p_seq.shape)
                correct_zs_per_h = []
                count_zs_per_h = count_z_per_h
                for iib in range(len(batch['spH_list'])):
                    valid_zs = find_valid_zs(batch['spH_prefix_list_info']['spH_prefix_list'][iib],
                                            batch['xy_seq_list_info'    ]['xy_seq_list'    ][iib],
                                            pad_token=tokens['pad'], predict_token=tokens['>'], comma_token=tokens[','])
                    if p_seq[iib] in valid_zs:
                        correct_zs_per_h.append(1)
                    else:
                        correct_zs_per_h.append(0)
                correct_zs_per_h = torch.tensor(correct_zs_per_h)

                batch_acc_zs.update(table_length_batch, correct_zs_per_h.data, count_zs_per_h)

            #pbar.set_description(f"{phase}-{icl_sampling} {len(strings[phase])} loss={batch_loss.avg[0]:.3f} acc_x={batch_acc_x.avg[0]:.3f} acc_y={batch_acc_y.avg[0]:.3f} acc_z={batch_acc_z.avg[0]:.3f}")
            pbar.set_description(f"{phase}-{icl_sampling} loss={batch_loss.avg[0]:.3f} acc_x={batch_acc_x.avg[0]:.3f} acc_y={batch_acc_y.avg[0]:.3f} acc_z={batch_acc_z.avg[0]:.3f} acc_zs={batch_acc_zs.avg[0]:.3f}")

        wandb_info={}
        for table_length in table_lengths:
            wandb_info[f"{phase}-{icl_sampling}{l2s(iid_probability)}/loss__{table_length}"] = batch_loss.avg[table_length]
            wandb_info[f"{phase}-{icl_sampling}{l2s(iid_probability)}/acc_x_{table_length}"] = batch_acc_x.avg[table_length]
            wandb_info[f"{phase}-{icl_sampling}{l2s(iid_probability)}/acc_y_{table_length}"] = batch_acc_y.avg[table_length]
            wandb_info[f"{phase}-{icl_sampling}{l2s(iid_probability)}/acc_z_{table_length}"] = batch_acc_z.avg[table_length]
            wandb_info[f"{phase}-{icl_sampling}{l2s(iid_probability)}/acc_h_{table_length}"] = batch_acc_h.avg[table_length]
            wandb_info[f"{phase}-{icl_sampling}{l2s(iid_probability)}/acc_s_{table_length}"] = batch_acc_s.avg[table_length]
            wandb_info[f"{phase}-{icl_sampling}{l2s(iid_probability)}/acc_zs_{table_length}"] = batch_acc_zs.avg[table_length]
        
    return wandb_info


if 1:
    hdata_hypers = 'split_based_on='+str(args.split_based_on) \
             +'_'+ 'num_training_tables='+str(args.num_training_tables) \
             +'_'+ 'num_x='+str(args.num_x) \
             +'_'+ 'num_y='+str(args.num_y) \
             +'_'+ 'sampling_disparity='+str(args.sampling_disparity) \
             +'_'+ 'icl_y_noise='+str(args.icl_y_noise) \
             +'_'+ 'random_seed='+str(args.random_seed)
    model_hypers = 'modelName='+str(args.modelName) \
             +'_'+ 'depth='+str(args.depth) \
             +'_'+ 'dim='+str(args.embed_dim) \
             +'_'+ 'heads='+str(args.num_heads)
    optim_hypers = 'lr='+str(args.lr) \
             +'_'+ 'wd='+str(args.wd) \
             +'_'+ 'BS='+str(args.batch_size) \
             +'_'+ 'Step='+str(args.n_steps) \
             +'_'+ 'EP='+str(args.epochs)

    # Initialize the data loader
    print(f'num_x={args.num_x}, split_based_on={args.split_based_on}, icl_k={args.icl_k}, n_steps={args.n_steps}')
    #hmanager = HypothesisManager(mode=args.mode, n=args.n, random_seed=args.random_seed, k=args.k, split_ratio=[0.6,0.3,0.1])
    

    num_x = args.num_x  # Number of elements in the mode
    random_seed = args.random_seed
    icl_k = args.icl_k  # Number of x-y pairs per sequence
    split_based_on = args.split_based_on  # 'hypothesis' or 'table'

    if split_based_on == 'table':
        if args.exp_name == 'TableLengthGeneralization':
            if args.num_x == 4:
                table_lengths = [4, 5, 6, 7]
                split_ratio = [0.7, 0.3]
                train_info = {4: 1274, 5: 1274, 6: 1274, 7:1274}  # Number of train tables to sample per length
                test__info = {4: 546, 5: 546, 6: 546, 7: 546}  # Number of train tables to sample per length
            if args.num_x == 5:
                table_lengths = [3, 4, 5, 6, 7]
                split_ratio = [2/3, 1/3]
                if args.num_training_tables != 0:
                    train_info = {
                        4: args.num_training_tables,
                        5: args.num_training_tables,
                        6: args.num_training_tables
                    }  # Number of train tables to sample per length
                else:
                    train_info = {4: 3000, 5: 3000, 6: 3000}  # Number of train tables to sample per length
                test__info = {3:1500, 4: 1500, 5: 1500, 6: 1500, 7: 1500}  # Number of train tables to sample per length
        elif args.exp_name == 'TableGeneralization':
            if args.num_x == 2:
                table_lengths = [2]
                split_ratio = [1, 0]  # Ratios for train and test splits
                train_info = {2: 6}  # Number of train tables to sample per length
                test__info = {2: 0}  # Number of test tables to sample per length
            if args.num_x == 3:
                table_lengths = [3]
                split_ratio = [56/56, 0/56]  # Ratios for train and test splits
                train_info = {3: 56}  # Number of train tables to sample per length
                test__info = {3: 0}  # Number of test tables to sample per length
            if args.num_x == 4:
                table_lengths = [4]
                split_ratio = [0.7, 0.3]  # Ratios for train and test splits
                if args.num_training_tables != 0:
                    train_info = {4: args.num_training_tables}  # Number of train tables to sample per length
                else:
                    train_info = {4: 1274}  # Number of train tables to sample per length
                test__info = {4: 526}  # Number of test tables to sample per length
            if args.num_x == 5:
                table_lengths = [4]
                split_ratio = [0.7, 0.3]  # Ratios for train and test splits
                train_info = {4: 1820}  # Number of train tables to sample per length
                test__info = {4: 1820}  # Number of test tables to sample per length
    
    if split_based_on == 'hypothesis':
        if args.exp_name == 'HypothesisGeneralization':
            if args.num_x == 4:
                raise Exception('Num of tables would be too small')
            if args.num_x == 5:
                table_lengths = [4]
                split_ratio = [0.5, 0.5]  # Ratios for train and test splits
                train_info = {4: 1820}  # Number of train tables to sample per length
                test__info = {4: 1820}  # Number of test tables to sample per length
        elif args.exp_name == 'HypothesisLengthGeneralization':
            if args.num_x == 6:
                table_lengths = [4, 5, 6, 7, 8]
                split_ratio = [2/3, 1/3]
                if args.num_training_tables != 0:
                    train_info = {
                        4: args.num_training_tables,
                        5: args.num_training_tables,
                        6: args.num_training_tables
                    }  # Number of train tables to sample per length
                else:
                    train_info = {5: 3000, 6: 3000, 7: 3000}  # Number of train tables to sample per length
                test__info = {4:1500, 5: 1500, 6: 1500, 7: 1500, 8: 1500}  # Number of train tables to sample per length
    if args.max_table_length < max(table_lengths):
        raise Exception('max_table_length too small')
    # if args.exp_name == 'table_length_basiccheck':
    #     if split_based_on != 'table':
    #         raise Exception('Wrong setting')
    #     if n == 4:
    #         table_lengths = [4, 5, 6, 7]
    #         split_ratio = [0.7, 0.3]
    #         train_info = {4: 1274, 5: 1274, 6: 1274, 7:1274}  # Number of train tables to sample per length
    #         test__info = {4: 546, 5: 546, 6: 546, 7: 546}  # Number of train tables to sample per length
    #     if n == 5:
    #         table_lengths = [3, 4, 5, 6, 7]
    #         split_ratio = [2/3, 1/3]
    #         train_info = {3: 3000, 4: 3000, 5: 3000, 6: 3000, 7: 3000}  # Number of train tables to sample per length
    #         test__info = {3:1500, 4: 1500, 5: 1500, 6: 1500, 7: 1500}  # Number of train tables to sample per length

    # if args.exp_name == 'table_length_generalization':
    #     if split_based_on != 'table':
    #         raise Exception('Wrong setting')
    #     if n == 4:
    #         table_lengths = [4, 5, 6, 7]
    #         split_ratio = [0.7, 0.3]
    #         train_info = {5: 1274, 6: 1274}  # Number of train tables to sample per length
    #         test__info = {4: 546, 5: 546, 6: 546, 7: 546}  # Number of train tables to sample per length
    #     if n == 5:
    #         table_lengths = [3, 4, 5, 6, 7]
    #         split_ratio = [2/3, 1/3]
    #         if args.num_training_tables != 0:
    #             train_info = {
    #                 4: args.num_training_tables,
    #                 5: args.num_training_tables,
    #                 6: args.num_training_tables
    #             }  # Number of train tables to sample per length
    #         else:
    #             train_info = {4: 3000, 5: 3000, 6: 3000}  # Number of train tables to sample per length
    #         test__info = {3:1500, 4: 1500, 5: 1500, 6: 1500, 7: 1500}  # Number of train tables to sample per length

    # Initialize the HypothesisManager

    iid_probability = generate_normalized_vector(args.num_x, args.sampling_disparity)

    hmanager = HypothesisManager(
        args,
        table_lengths=table_lengths,
        split_ratio=split_ratio,
        train_info=train_info,
        test__info=test__info
    )
    train_dmanager = DataloaderManager(
        args,
        hmanager = hmanager,
        n_steps = args.n_steps,
        split = 'train',
        preshuffle = True,
        icl_sampling = args.icl_sampling,
        iid_probability = iid_probability,
        icl_y_noise = args.icl_y_noise
    )
    test__dmanager = DataloaderManager(
        args,
        hmanager = hmanager,
        n_steps = 1024,
        split = 'test_',
        preshuffle = True,
        icl_sampling = args.icl_sampling,
        iid_probability = iid_probability,
        icl_y_noise = args.icl_y_noise
    )
    opt___dmanager = DataloaderManager(
        args,
        hmanager = hmanager,
        n_steps = 1024,
        split = 'test_',
        preshuffle = True,
        icl_sampling = 'optimal'
    )


    # wandb
    if args.wandb:
        wandb.login(key='0e030fcc130348fb3127f6140ac82c773fa4b4d9')
        # if args.method in ['normal', 'mix']:
        #     name = f'model={args.modelName} h_prefix={1} method={args.method} k={args.k}'
        # if args.method == 'optimal':
        #     name = f'model={args.modelName} h_prefix={1} method={args.method}'
        name = f'modelName={args.modelName}'
        run = wandb.init(
            # Set the project where this run will be logged
            project= f'{args.HEAD} {args.exp_name} icl={args.icl_sampling} num_x={args.num_x}',
            name = name,
            entity = 'myhakureimu',
            dir='../wandb',
            # Track hyperparameters and run metadata
            config={
                'seed': args.random_seed,
                'num_x': args.num_x,
                'num_y': args.num_y,
                'max_table_length': args.max_table_length,
                
                'icl_k': args.icl_k,
                'loss_on': args.loss_on,
                'icl_sampling': args.icl_sampling,
                'sampling_disparity': args.sampling_disparity,
                'icl_y_noise': args.icl_y_noise,
                'h_prefix_format': args.h_prefix_format,
                'mix_prob_train1': args.mix_prob_train1,

                'num_training_tables': args.num_training_tables,

                'modelName': args.modelName,
                'num_heads': args.num_heads,
                'depth': args.depth,
                'embed_dim': args.embed_dim,
                'llm_max_length': args.llm_max_length,

                'split_based_on': split_based_on,
                'split_ratio': split_ratio,
                'table_lengths': table_lengths,
                'train_info': train_info,
                'test__info': test__info,

                'lr': args.lr,
                'wd': args.wd,
                'batch_size': args.batch_size,
                'n_steps': args.n_steps,
                'epochs': args.epochs
            },
        )
        wandb.define_metric("*", step_metric="global_step")
    
    # folder
    print('***** ' + hdata_hypers + ' *****')
    print('***** ' + model_hypers + ' *****')
    print('***** ' + optim_hypers + ' *****')
    folder = 'saved/'+args.HEAD+args.exp_name+'/'+hdata_hypers+'/'+model_hypers+'/'+optim_hypers+'/'
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # model
    if args.modelName == 'dual': #dual
        model = TransformerModel(
            n_dims = hmanager.num_tokens,
            n_positions = args.llm_max_length, 
            n_embd = args.embed_dim,
            n_layer = args.depth, 
            n_head = args.num_heads
        )
    if args.modelName == 'mamba': #dual
        model = KLayerMambaModel(
            num_tokens = hmanager.num_tokens,
            num_layer = args.depth,
            d_model = args.embed_dim,
            d_state = 16,
            d_conv = 4,
            expand = 2,
        )
    if args.modelName == 'lstm':
        model = LSTMModel(
            input_dim = hmanager.num_tokens, 
            hidden_dim = args.embed_dim,
            output_dim = hmanager.num_tokens,
            num_layers = args.depth
        )
    if args.modelName == 'gru':
        model = GRUModel(
            input_dim = hmanager.num_tokens, 
            hidden_dim = args.embed_dim,
            output_dim = hmanager.num_tokens, 
            num_layers = args.depth
        )
    if args.modelName == 'nano': #nanoGPT
        config = GPTConfig(
            input_dim = hmanager.num_tokens,
            block_size = args.llm_max_length,
            #vocab_size = 50304, # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
            n_layer = args.depth,
            n_head = args.num_heads,
            n_embd = args.embed_dim,
            bias = True, # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
        )
        model = NanoGPT(config)
    if args.modelName == 'pytorch':
        model = PytorchTransformer(
            i_dimensions = hmanager.num_tokens, 
            h_dimensions = args.embed_dim, 
            o_dimensions = hmanager.num_tokens, 
            num_layers = args.depth, 
            num_heads = args.num_heads,
            max_seq_length = args.llm_max_length)

    model.cuda()
    #total_params = sum(p.numel() for p in model._read_in.parameters())
    #print(f"Total number of parameters: {total_params}") 
    total_params = sum(p.numel() for p in model._backbone.parameters())
    print(f"_backbone: {total_params}")
    total_params = sum(p.numel() for p in model._backbone.wte.parameters())
    total_params_1 = total_params
    print(f"_backbone.wte: {total_params}")
    total_params = sum(p.numel() for p in model._backbone.wpe.parameters())
    total_params_2 = total_params
    print(f"_backbone.wpe: {total_params}")
    total_params = sum(p.numel() for p in model._backbone.h.parameters())
    total_params_3 = total_params
    print(f"_backbone.h: {total_params}")
    total_params = sum(p.numel() for p in model._backbone.ln_f.parameters())
    total_params_4 = total_params
    print(f"_backbone.ln_f: {total_params}")
    print(total_params_1+total_params_2+total_params_3+total_params_4)
    #total_params = sum(p.numel() for p in model._read_out.parameters())
    #print(f"Total number of parameters: {total_params}")
    #print(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas = (0.9, 0.999))
    
    
    # print('******** EP = ' +str(0)+ ' / ' +str(args.epochs)+ ' *******')
    # epoch = 0
    # split = 'test'
    # wandb_valid_info = train_model(args, split, hmanager, model, optimizer, epoch=epoch)
    # if args.wandb:
    #     wandb_valid_info['global_step'] = epoch
    #     wandb.log(wandb_valid_info)
    epoch = 0
    if epoch%16 == 0:
        phase = 'test_'
        wandb_test1_info = train_model(args, phase, table_lengths, test__dmanager, model, optimizer, epoch=epoch)
        phase = 'test_'
        wandb_test2_info = train_model(args, phase, table_lengths, opt___dmanager, model, optimizer, epoch=epoch)
    else:
        wandb_test1_info = {}
        wandb_test2_info = {}

    load_from = 0
    for epoch in range(2, args.epochs+1):
        last_file = folder + f'EP={epoch-1}'
        curr_file = folder + f'EP={epoch}'
        last_exists = os.path.exists(last_file)
        curr_exists = os.path.exists(curr_file)
        if last_exists and curr_exists:
            load_from = epoch-1

    if load_from != 0:
        save_path = folder + f'EP={load_from}'
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    for epoch in range(load_from+1, args.epochs+1):
        print('******** EP = ' +str(epoch)+ ' / ' +str(args.epochs)+ ' *******')
        #print(model._read_out.weight.data)
        #print(table_lengths)
        if 1:#epoch!=0: #train
            phase = 'train'
            wandb_train_info = train_model(args, phase, table_lengths, train_dmanager, model, optimizer, epoch=epoch)

        if epoch%16 == 0:
            phase = 'test_'
            wandb_test1_info = train_model(args, phase, table_lengths, test__dmanager, model, optimizer, epoch=epoch)
            phase = 'test_'
            wandb_test2_info = train_model(args, phase, table_lengths, opt___dmanager, model, optimizer, epoch=epoch)
        else:
            wandb_test1_info = {}
            wandb_test2_info = {}
        
        # Combine all metrics into one dictionary
        combined_metrics = {}
        combined_metrics.update(wandb_train_info)
        combined_metrics.update(wandb_test1_info)
        combined_metrics.update(wandb_test2_info)
        if args.wandb:
            combined_metrics['global_step'] = epoch
            wandb.log(combined_metrics, step=epoch)

        save_path = folder + f'EP={epoch}'
        print(save_path)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_path)
        if epoch-2 >= 0:
            dele_path = folder + f'EP={epoch-2}'
            if os.path.exists(dele_path):
                os.remove(dele_path)
        
        # import pickle 

        # with open('strings.pkl', 'wb') as f:
        #     pickle.dump(strings, f)

        if 0:
            phase = 'test_'
            wandb_valid_info = train_model(args, phase, table_lengths, opt___dmanager, model, optimizer, epoch=epoch)
            if args.wandb:
                wandb_valid_info['global_step'] = epoch
                wandb.log(wandb_valid_info, step=epoch, commit=False)
            
          