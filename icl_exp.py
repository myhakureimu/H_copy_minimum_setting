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
import pickle 
import torch.nn.functional as F

from new_hmanager import HypothesisManager, DataloaderManager
from find_valid_zs import find_valid_zs
from get_config import get_config
from InverseSqrtWithWarmupLR import InverseSqrtWithWarmupLR

matplotlib.rc('text', usetex=True)
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')
# python train.py --name=GPT

# environment setting
parser = argparse.ArgumentParser(description='PyTorch In-context Learning Training Code')
parser.add_argument('--gpu', default='0', type=str, help='which gpus to use')
parser.add_argument('--wandb', default=0, type=int)

parser.add_argument('--HEAD', default='Diversity', type=str)
parser.add_argument('--exp_name', default='IOHypothesis', type=str)
parser.add_argument('--training_content', default='h+xy', choices = ['h+xy+z', 'h+xy', 'xy'])
#arxived args
# parser.add_argument('--SigmaRe', default=2, type=int)
# parser.add_argument('--NormAtt', default=0, type=int)
# parser.add_argument('--FirstLayerNorm', default=1, type=int)


# H setting for init hypothesismanager
''' parser.add_argument('--mode', default='binary', type=str, choices=['binary', 'permutation'])  #binary only '''
parser.add_argument('--num_x', default=6, type=int)
parser.add_argument('--num_y', default=2, type=int)
parser.add_argument('--num_training_hypotheses', default=8, type=int)
parser.add_argument('--num_training_tables', default=1, type=int)
parser.add_argument('--max_table_length', default=8, type=int)
# table_lengths
#parser.add_argument('--split_based_on', default='table', type=str)
parser.add_argument('--random_seed', default=1, type=int, help='the seed used for torch & numpy')
# split_ratio
# train_info
# test__info

# H+ICL format for dataloadermanager
parser.add_argument('--icl_k', default=12, type=int)
parser.add_argument('--loss_on', default='all', type=str, choices=['all', 'icl&>z', 'y&z', 'z'], help = 'all=prefix&icl&z, icl=x&y&>')
parser.add_argument('--icl_sampling', default='iid', type=str, choices = ['ordered', 'shuffle', 'iid', 'optimal', 'mix'])
parser.add_argument('--sampling_disparity', default=1.0, type=float)
parser.add_argument('--icl_y_noise', default=0.0, type=float)
parser.add_argument('--h_prefix_format', default=0, type=int, choices=[0,1])
parser.add_argument('--mix_prob_train1', default=0.5, type=float)


# model setting
parser.add_argument('--modelName', default='transformer', type=str, choices=['transformer', 'mamba', 'lstm', 'gru'])
parser.add_argument('--depth', default=8, type=int, help='depth of the transformer architecture (default: 12)')
parser.add_argument('--embed_dim', default=128, type=int, help='embedding dimension of the transformer feature extractor (default: 256)')
parser.add_argument('--num_heads', default=2, type=int, help='number of heads for multi-headed attention (default: 8)')
# parser.add_argument('--dropout', default=0.0, type=float, help='dropout')
parser.add_argument('--llm_max_length', default=512, type=int, help='maximum sequence length of the input (default: 11)')

#optimization
parser.add_argument('--lr', default=0.00002, type=float, help='initial model learning rate') #0.0005
parser.add_argument('--use_scheduler', default=1, type=int, choices=[0,1])
parser.add_argument('--wd', default=0.0005, type=float, help='weight decay hyperparameter (default: 0.00001)') #0.1
parser.add_argument('--batch_size', default=16, type=int, help='mini-batch size (default: 64)') #32
parser.add_argument('--n_steps', default=512, type=int, help='total number of training steps we want to run')
parser.add_argument('--epochs', default=512, type=int, help='number of total epochs to run')
parser.add_argument('--epochs2test', default=32, type=int, help='number of epochs to test')

parser.set_defaults(augment=True)
args = parser.parse_args()

args.n_steps = int(1024 * 16 / args.batch_size)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
if args.HEAD == 'FourGeneralization':
    setproctitle.setproctitle(f'{args.exp_name} {args.modelName} {args.random_seed}')
if args.HEAD == 'HyperSearch':
    setproctitle.setproctitle(f'{args.exp_name} {args.use_scheduler} {args.lr}')
if args.HEAD == 'NUMTRAIN':
    setproctitle.setproctitle(f'{args.exp_name} {args.modelName} {args.num_training_tables} {args.random_seed}')
if args.HEAD == 'ICL':
    setproctitle.setproctitle(f'{args.exp_name} {args.training_content} {args.random_seed}')
if args.HEAD == 'DP':
    setproctitle.setproctitle(f'{args.exp_name} {args.sampling_disparity} {args.random_seed}')
if args.HEAD == 'Diversity':
    setproctitle.setproctitle(f'{args.exp_name} {args.training_content} {args.num_training_hypotheses} {args.random_seed}')

if args.HEAD == 'FourGeneralization':
    name = f'model={args.modelName} seed={args.random_seed}'
if args.HEAD == 'HyperSearch':
    name = f'lr={args.lr} scheduler={args.use_scheduler} seed={args.random_seed}'
if args.HEAD == 'NUMTRAIN':
    name = f'model={args.modelName} num={args.num_training_tables} seed={args.random_seed}'
if args.HEAD == 'ICL':
    name = f'content={args.training_content} seed={args.random_seed}'
if args.HEAD == 'DP':
    name = f'DP={args.sampling_disparity} seed={args.random_seed}'
if args.HEAD == 'Diversity':
    name = f'content={args.training_content} num={args.num_training_hypotheses} seed={args.random_seed}'

import torch
import torch.nn as nn
if args.modelName == 'nano':
    from utils.nano_gpt import GPTConfig, NanoGPT
if args.modelName == 'pytorch':
    from utils.pytorch_transformer import PytorchTransformer
if args.modelName == 'transformer':
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
def traintest_model(args, phase, table_lengths, dmanager, model, optimizer, epoch):
    wandb_info = {}
    if phase not in ['train', 'testI', 'testO']:
        raise Exception(f'phase = {phase} is not valid')
    icl_sampling = dmanager.icl_sampling
    iid_probability = dmanager.iid_probability
    dataloader = dmanager.get_pytorch_dataloader()
    int2token = dmanager.hmanager.int2token
    tokens = dmanager.tokens
    #print(int2token)

    loss_f = torch.nn.CrossEntropyLoss(reduction = 'none')
    
    batch_loss = AverageMeter(table_lengths)
    batch_numx = AverageMeter(table_lengths) #check how many xs in the query
    batch_acc_x = AverageMeter(table_lengths)
    batch_acc_y = AverageMeter(table_lengths)
    batch_acc_icl = AverageMeter(list(np.arange(1, args.icl_k+1)))
    batch_acc_z = AverageMeter(table_lengths)
    batch_acc_h = AverageMeter(table_lengths)
    batch_acc_s = AverageMeter(table_lengths)
    batch_acc_zs = AverageMeter(table_lengths)
    if phase == 'train':
        model.train()
    if phase in ['testI','testO']:
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

            correct_xmask = torch.cat([spH_prefix_xmask, xy_seq_xmask, z_suffix_xmask], dim=1).to(torch.float32)
            correct_ymask = torch.cat([spH_prefix_ymask, xy_seq_ymask, z_suffix_ymask], dim=1).to(torch.float32)
            correct_zmask = torch.cat([spH_prefix_zmask, xy_seq_zmask, z_suffix_zmask], dim=1).to(torch.float32)
            correct_hmask = torch.cat([spH_prefix_hmask, xy_seq_hmask, z_suffix_hmask], dim=1).to(torch.float32)
            correct_smask = torch.cat([spH_prefix_smask, xy_seq_smask, z_suffix_smask], dim=1).to(torch.float32)
            icl_y_mask    = torch.cat([torch.zeros_like(spH_prefix), xy_seq_ymask, torch.zeros_like(z_suffix_zmask)], dim=1).to(torch.float32)
            
            all_seq = torch.cat([spH_prefix, xy_seq, z_suffix], dim=1)
            #print(all_seq.shape)
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
            
            # process training content
            cut_h  = spH_prefix.shape[1]
            cut_xy = xy_seq    .shape[1]
            cut_z  = z_suffix  .shape[1]
            if args.training_content == 'h+xy+z':
                start = 0
                end   = cut_h + cut_xy + cut_z
            elif args.training_content == 'h+xy':
                start = 0
                end   = cut_h + cut_xy
            elif args.training_content == 'xy':
                start = cut_h
                end   = cut_h + cut_xy



            all_seq     = all_seq    [:, start:end]
            losses_mask = losses_mask[:, start:end]
            correct_xmask = correct_xmask[:, start:end]
            correct_ymask = correct_ymask[:, start:end]
            correct_zmask = correct_zmask[:, start:end]
            correct_hmask = correct_hmask[:, start:end]
            correct_smask = correct_smask[:, start:end]
            icl_y_mask    = icl_y_mask   [:, start:end]

            i_seq = all_seq[:, :-1]
            o_seq = all_seq[:, 1:]
            losses_mask = losses_mask[:, 1:]
            correct_xmask = correct_xmask[:, 1:]
            correct_ymask = correct_ymask[:, 1:]
            correct_zmask = correct_zmask[:, 1:]
            correct_hmask = correct_hmask[:, 1:]
            correct_smask = correct_smask[:, 1:]
            icl_y_mask    = icl_y_mask   [:, 1:]
            
            p_seq = model.forward(i_seq)
            losses = loss_f(p_seq.transpose(1, 2), o_seq.to(torch.long))
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

                ##correct_x_per_h = torch.sum(correct*correct_xmask, dim=1)
                ##correct_y_per_h = torch.sum(correct*correct_ymask, dim=1)
                correct_z_per_h = torch.sum(correct*correct_zmask, dim=1)
                ##correct_h_per_h = torch.sum(correct*correct_hmask, dim=1)
                ##correct_s_per_h = torch.sum(correct*correct_smask, dim=1)
                correct_icl_y   = torch.mean(1.0*(correct[icl_y_mask==1]).reshape([args.batch_size,-1]), dim=0)
                
                ##count_x_per_h = torch.sum(correct_xmask, dim=1)
                ##count_y_per_h = torch.sum(correct_ymask, dim=1)
                count_z_per_h = torch.sum(correct_zmask, dim=1)
                ##count_h_per_h = torch.sum(correct_hmask, dim=1)
                ##count_s_per_h = torch.sum(correct_smask, dim=1)
                count_icl_y   = torch.ones_like(correct_icl_y)
                icl_pos = [pos+1 for pos in range(args.icl_k)]

                ###xs = torch.stack([a[b == 1] for a, b in zip(xy_seq, xy_seq_xmask)], dim=0)
                ###one_hot_result = F.one_hot(torch.tensor([torch.unique(row).size(0) - 1 for row in xs]), num_classes=args.num_x).cuda()

            ###batch_numx .update(table_length_batch, one_hot_result .data, count_z_per_h)

            batch_loss .update(table_length_batch, loss_per_h     .data, count_per_h)
            ##batch_acc_x.update(table_length_batch, correct_x_per_h.data, count_x_per_h)
            ##batch_acc_y.update(table_length_batch, correct_y_per_h.data, count_y_per_h)
            batch_acc_icl.update(icl_pos         , correct_icl_y  .data, count_icl_y)
            
            batch_acc_z.update(table_length_batch, correct_z_per_h.data, count_z_per_h)
            ##batch_acc_h.update(table_length_batch, correct_h_per_h.data, count_h_per_h)
            ##batch_acc_s.update(table_length_batch, correct_s_per_h.data, count_s_per_h)

            # if (phase in ['testI', 'testO']) and (args.training_content == 'h+xy+z'):
            #     p_seq = torch.argmax(p_seq, dim=2)[correct_zmask==1]
            #     correct_zs_per_h = []
            #     count_zs_per_h = count_z_per_h
            #     for iib in range(len(batch['spH_list'])):
            #         valid_zs = find_valid_zs(batch['spH_prefix_list_info']['spH_prefix_list'][iib],
            #                                 batch['xy_seq_list_info'    ]['xy_seq_list'    ][iib],
            #                                 pad_token=tokens['pad'], predict_token=tokens['>'], comma_token=tokens[','])
            #         if p_seq[iib] in valid_zs:
            #             correct_zs_per_h.append(1)
            #         else:
            #             correct_zs_per_h.append(0)
            #     correct_zs_per_h = torch.tensor(correct_zs_per_h)
            #     batch_acc_zs.update(table_length_batch, correct_zs_per_h.data, count_zs_per_h)

            pbar.set_description(f"{phase}-{icl_sampling} acc_z={batch_acc_z.avg[0]:.3f}")
            #pbar.set_description(f"{phase}-{icl_sampling} loss={batch_loss.avg[0]:.3f} acc_x={batch_acc_x.avg[0]:.3f} acc_y={batch_acc_y.avg[0]:.3f} acc_z={batch_acc_z.avg[0]:.3f} acc_zs={batch_acc_zs.avg[0]:.3f}")

        ###print(batch_numx.avg[0])

        wandb_info={}
        for table_length in table_lengths:
            wandb_info[f"{phase}-{icl_sampling}{l2s(iid_probability)}/loss__{table_length}"] = batch_loss.avg[table_length]
            ##wandb_info[f"{phase}-{icl_sampling}{l2s(iid_probability)}/acc_x_{table_length}"] = batch_acc_x.avg[table_length]
            ##wandb_info[f"{phase}-{icl_sampling}{l2s(iid_probability)}/acc_y_{table_length}"] = batch_acc_y.avg[table_length]
            wandb_info[f"{phase}-{icl_sampling}{l2s(iid_probability)}/acc_z_{table_length}"] = batch_acc_z.avg[table_length]
            ##wandb_info[f"{phase}-{icl_sampling}{l2s(iid_probability)}/acc_h_{table_length}"] = batch_acc_h.avg[table_length]
            ##wandb_info[f"{phase}-{icl_sampling}{l2s(iid_probability)}/acc_s_{table_length}"] = batch_acc_s.avg[table_length]
            ##wandb_info[f"{phase}-{icl_sampling}{l2s(iid_probability)}/acc_zs_{table_length}"] = batch_acc_zs.avg[table_length]
        for pos in range(args.icl_k):
            wandb_info[f"{phase}-{icl_sampling}{l2s(iid_probability)}_icl/pos{pos+1}"] = batch_acc_icl.avg[pos+1]
        

    return wandb_info


if 1:
    # num_x = args.num_x  # Number of elements in the mode
    # random_seed = args.random_seed
    # icl_k = args.icl_k  # Number of x-y pairs per sequence
    
    table_lengths, num_IO_h, train_info, testI_info, testO_info = get_config(args)

    if args.max_table_length < max(table_lengths):
        raise Exception('max_table_length too small')

    # Initialize the HypothesisManager
    iid_probability = generate_normalized_vector(args.num_x, args.sampling_disparity)
    hmanager = HypothesisManager(
        args,
        table_lengths=table_lengths,
        num_IO_h=num_IO_h,
        train_info=train_info,
        testI_info=testI_info,
        testO_info=testO_info,
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
    testI_dmanager = DataloaderManager(
        args,
        hmanager = hmanager,
        n_steps = args.n_steps,
        split = 'testI',
        preshuffle = True,
        icl_sampling = args.icl_sampling,
        iid_probability = iid_probability,
        icl_y_noise = args.icl_y_noise
    )
    testO_dmanager = DataloaderManager(
        args,
        hmanager = hmanager,
        n_steps = args.n_steps,
        split = 'testO',
        preshuffle = True,
        icl_sampling = args.icl_sampling,
        iid_probability = iid_probability,
        icl_y_noise = args.icl_y_noise
    )
    opt_I_dmanager = DataloaderManager(
        args,
        hmanager = hmanager,
        n_steps = args.n_steps,
        split = 'testI',
        preshuffle = True,
        icl_sampling = 'optimal'
    )
    opt_O_dmanager = DataloaderManager(
        args,
        hmanager = hmanager,
        n_steps = args.n_steps,
        split = 'testO',
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
        #name = f'content={args.training_content} sparsity={args.num_training_hypotheses} seed={args.random_seed}'
        run = wandb.init(
            # Set the project where this run will be logged
            project= f'{args.HEAD} {args.exp_name} numx={args.num_x}',
            name = name,
            entity = 'myhakureimu',
            dir='../wandb',
            # Track hyperparameters and run metadata
            config={
                'HEAD': args.HEAD,
                'exp_name': args.exp_name,
                'training_content': args.training_content,
                
                'num_x': args.num_x,
                'num_y': args.num_y,
                'num_training_hypotheses': args.num_training_hypotheses,
                'num_training_tables': args.num_training_tables,
                'max_table_length': args.max_table_length,
                #'split_based_on': args.split_based_on,
                'random_seed': args.random_seed,

                'icl_k': args.icl_k,
                'loss_on': args.loss_on,
                'icl_sampling': args.icl_sampling,
                'sampling_disparity': args.sampling_disparity,
                'icl_y_noise': args.icl_y_noise,
                'h_prefix_format': args.h_prefix_format,
                'mix_prob_train1': args.mix_prob_train1,

                'modelName': args.modelName,
                'depth': args.depth,
                'embed_dim': args.embed_dim,
                'num_heads': args.num_heads,
                'llm_max_length': args.llm_max_length,

                'lr': args.lr,
                'use_scheduler': args.use_scheduler,
                'wd': args.wd,
                'batch_size': args.batch_size,
                'n_steps': args.n_steps,
                'epochs': args.epochs,
                'epochs2test': args.epochs2test,
                #'split_ratio': split_ratio,
                'table_lengths': table_lengths,
                'num_IO_h': num_IO_h,
                'train_info': train_info,
                'testI_info': testI_info,
                'testO_info': testO_info,
            },
        )
        wandb.define_metric("*", step_metric="global_step")
    

    hmanager_hypers = 'num_x='+str(args.num_x) \
             +'_'+ 'num_y='+str(args.num_y) \
             +'_'+ 'num_training_hypotheses='+str(args.num_training_hypotheses) \
             +'_'+ 'num_training_tables='+str(args.num_training_tables) \
             +'_'+ 'max_table_length='+str(args.max_table_length) \
             +'_'+ 'random_seed='+str(args.random_seed)
    dataloader_hypers = 'icl_k='+str(args.icl_k) \
             +'_'+ 'loss_on='+str(args.loss_on) \
             +'_'+ 'icl_sampling='+str(args.icl_sampling) \
             +'_'+ 'sampling_disparity='+str(args.sampling_disparity) \
             +'_'+ 'icl_y_noise='+str(args.icl_y_noise) \
             +'_'+ 'h_prefix_format='+str(args.h_prefix_format) \
             +'_'+ 'mix_prob_train1='+str(args.mix_prob_train1)
    model_hypers = 'modelName='+str(args.modelName) \
             +'_'+ 'depth='+str(args.depth) \
             +'_'+ 'dim='+str(args.embed_dim) \
             +'_'+ 'heads='+str(args.num_heads) \
             +'_'+ 'llm_max_length'+str(args.llm_max_length)
    optim_hypers = 'lr='+str(args.lr) \
             +'_'+ 'use_scheduler='+str(args.use_scheduler) \
             +'_'+ 'wd='+str(args.wd) \
             +'_'+ 'BS='+str(args.batch_size) \
             +'_'+ 'Step='+str(args.n_steps) \
             +'_'+ 'EP='+str(args.epochs)


    # folder
    print('***** ' + hmanager_hypers + ' *****')
    print('***** ' + dataloader_hypers + ' *****')
    print('***** ' + model_hypers + ' *****')
    print('***** ' + optim_hypers + ' *****')
    folder = 'saved' \
        +'/'+args.HEAD+'_'+args.exp_name+'_'+args.training_content \
        +'/'+hmanager_hypers+'/'+dataloader_hypers+'/'+model_hypers+'/'+optim_hypers+'/'
    pkl_folder = 'saved4plot' \
        +'/'+args.HEAD \
        +'/'+args.exp_name+'_'+args.training_content \
        +'/'+ name

    if not os.path.exists(folder):
        os.makedirs(folder)
    if not os.path.exists(pkl_folder):
        os.makedirs(pkl_folder)

    # model
    if args.modelName == 'transformer': #dual
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
    # total_params = sum(p.numel() for p in model._backbone.parameters())
    # print(f"_backbone: {total_params}")
    # total_params = sum(p.numel() for p in model._backbone.wte.parameters())
    # total_params_1 = total_params
    # print(f"_backbone.wte: {total_params}")
    # total_params = sum(p.numel() for p in model._backbone.wpe.parameters())
    # total_params_2 = total_params
    # print(f"_backbone.wpe: {total_params}")
    # total_params = sum(p.numel() for p in model._backbone.h.parameters())
    # total_params_3 = total_params
    # print(f"_backbone.h: {total_params}")
    # total_params = sum(p.numel() for p in model._backbone.ln_f.parameters())
    # total_params_4 = total_params
    # print(f"_backbone.ln_f: {total_params}")
    # print(total_params_1+total_params_2+total_params_3+total_params_4)
    #total_params = sum(p.numel() for p in model._read_out.parameters())
    #print(f"Total number of parameters: {total_params}")
    #print(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas = (0.9, 0.999))
    if args.use_scheduler == True:
        scheduler = InverseSqrtWithWarmupLR(optimizer, warmup_epochs=64, base_lr=args.lr)

    # print('******** EP = ' +str(0)+ ' / ' +str(args.epochs)+ ' *******')
    epoch = 0
    if epoch%args.epochs2test == 0:
        phase = 'testI'
        wandb_test1_info = traintest_model(args, phase, table_lengths, opt_I_dmanager, model, optimizer, epoch=epoch)
        wandb_test2_info = traintest_model(args, phase, table_lengths, testI_dmanager, model, optimizer, epoch=epoch)
        phase = 'testO'
        wandb_test3_info = traintest_model(args, phase, table_lengths, opt_O_dmanager, model, optimizer, epoch=epoch)
        wandb_test4_info = traintest_model(args, phase, table_lengths, testO_dmanager, model, optimizer, epoch=epoch)
    else:
        wandb_test1_info = {}
        wandb_test2_info = {}
        wandb_test3_info = {}
        wandb_test4_info = {}

    # Combine all metrics into one dictionary
    combined_metrics = {}
    combined_metrics.update(wandb_test1_info)
    combined_metrics.update(wandb_test2_info)
    combined_metrics.update(wandb_test3_info)
    combined_metrics.update(wandb_test4_info)
    
    if args.wandb:
        combined_metrics['global_step'] = epoch
        wandb.log(combined_metrics, step=epoch)
        with open(f'{pkl_folder}/{epoch}.pkl', 'wb') as f:
            pickle.dump(combined_metrics, f)

    for epoch in range(1, args.epochs+1):
        print('******** EP = ' +str(epoch)+ ' / ' +str(args.epochs)+ ' *******')
        #print(model._read_out.weight.data)
        #print(table_lengths)
        if args.use_scheduler:
            scheduler.step()
        if 1:#epoch!=0: #train
            phase = 'train'
            wandb_train_info = traintest_model(args, phase, table_lengths, train_dmanager, model, optimizer, epoch=epoch)
        if epoch%args.epochs2test == 0:
            phase = 'testI'
            wandb_test1_info = traintest_model(args, phase, table_lengths, testI_dmanager, model, optimizer, epoch=epoch)
            wandb_test2_info = traintest_model(args, phase, table_lengths, opt_I_dmanager, model, optimizer, epoch=epoch)
            phase = 'testO'
            wandb_test3_info = traintest_model(args, phase, table_lengths, testO_dmanager, model, optimizer, epoch=epoch)
            wandb_test4_info = traintest_model(args, phase, table_lengths, opt_O_dmanager, model, optimizer, epoch=epoch)
        else:
            wandb_test1_info = {}
            wandb_test2_info = {}
            wandb_test3_info = {}
            wandb_test4_info = {}
        
        # Combine all metrics into one dictionary
        combined_metrics = {}
        combined_metrics.update(wandb_train_info)
        combined_metrics.update(wandb_test1_info)
        combined_metrics.update(wandb_test2_info)
        combined_metrics.update(wandb_test3_info)
        combined_metrics.update(wandb_test4_info)

        if args.wandb:
            combined_metrics['global_step'] = epoch
            combined_metrics['lr'] = optimizer.param_groups[0]['lr']
            wandb.log(combined_metrics, step=epoch)
            if epoch%args.epochs2test == 0:
                with open(f'{pkl_folder}/{epoch}.pkl', 'wb') as f:
                    pickle.dump(combined_metrics, f)
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