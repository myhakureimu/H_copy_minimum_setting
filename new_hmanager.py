import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import itertools
import math
import random
import time
from tqdm import tqdm
from construct_shuffled_nanned_table import construct_shuffled_nanned_table
from efficient_sampling_combinations import sample_random_combinations


def repeat_list_to_length(lst, K):
    # Repeat the list as many times as needed and slice to exactly K elements
    repeated_list = (lst * (K // len(lst) + 1))[:K]
    return repeated_list


class HypothesisManager:
    def __init__(self, args, table_lengths, num_IO_h, train_info, testI_info, testO_info):
        self.h_prefix_format = args.h_prefix_format
        self.random_seed = args.random_seed

        #self.mode = 'binary'
        self.num_x = args.num_x
        self.num_y = args.num_y
        self.max_table_length = args.max_table_length
        self.table_lengths = table_lengths

        #self.split_based_on = args.split_based_on
        self.num_IO_h = num_IO_h
        self.train_info = train_info
        self.testI_info = testI_info
        self.testO_info = testO_info

        self.max_num_tables = 2**14
        """
        Initializes the HypothesisManager with the specified parameters.

        Parameters:
        - random_seed (int): Random seed for reproducibility.

        - mode (str): 'permutation', 'D2Dmapping', or 'binary'.
        - n (int): Number of elements in the permutations.
        - max_table_length (int): max_table_length.
        - table_lengths (list): List of table lengths (number of hypotheses per table).

        - k (int): Number of x-y pairs per sequence.

        - split_ratio (list): Ratios for train and test splits, e.g., [0.7, 0.3].
        - split_based_on (str): 'hypothesis' or 'table'.
        - train_info (dict): Mapping table lengths to number of tables for training.
        - test__info (dict): Mapping table lengths to number of tables for testing.
        """
        
        self._init_tokens()
        print(f'***** self.tokens *****')
        print(self.tokens)
        self.num_tokens = sum(len(token_list) if isinstance(token_list, list) else 1 for token_list in self.tokens.values())


        # Generate all possible hypotheses
        self._generate_all_hypotheses()
        print(f'***** self.all_hypotheses *****')
        for i in range(3):
            print(self.all_hypotheses[i])
        print(f'***** self.num_all_h *****')
        print(f'= {self.num_all_h}')

        # Initialize the tables dictionaries

        # Split the data according to split_based_on
        # if self.split_based_on == 'hypothesis':
        #     self._split_based_on_hypotheses()
        # elif self.split_based_on == 'table':
        #     self._split_based_on_tables()
        # else:
        #     raise ValueError("split_based_on must be 'hypothesis' or 'table'.")
        self.I_h_indices, self.O_h_indices = {}, {}
        self._split_hypotheses_to_IO()
        self.I_tables, self.O_tables = {}, {}
        self._construct_max_num_tables_and_shuffle()

        print('num I_tables:')
        for key, value in self.I_tables.items():
            print(key, len(value))
        print('num O_tables:')
        for key, value in self.O_tables.items():
            print(key, len(value))

        # Further sample train and test tables according to train_info and test__info
        self.train_tables = {}
        self.testI_tables = {}
        self.testO_tables = {}
        self._sample_traintest_tables()

    def _init_tokens(self):
        # set token values
        self.tokens = {}
        token_index = 0

        self.tokens['xs'] = []  # [0, num_x-1]
        for i in range(self.num_x):
            self.tokens['xs'].append(token_index)
            token_index += 1
        
        self.tokens['ys'] = []  # [num_x, num_y+num_x-1]
        for i in range(self.num_y):
            self.tokens['ys'].append(token_index)
            token_index += 1

        self.tokens['pad'] = token_index # [num_y+num_x]
        token_index += 1
        self.tokens['nan'] = token_index # [num_y+num_x+1]
        token_index += 1
        self.tokens['>'] = token_index # [num_y+num_x+1]
        token_index += 1
        self.tokens[','] = token_index # [num_y+num_x+1]
        token_index += 1
        if self.h_prefix_format == 1:
            self.tokens[':'] = token_index # [num_y+num_x+1]
            token_index += 1

        self.tokens['hH_Z'] = [] # [num_y+num_x+2, num_y+num_x+1+max_table_length]
        for i in range(self.max_table_length):
            self.tokens[f'hH_Z'].append(token_index)
            token_index += 1
        
        self.int2token = {}
        for k, v in self.tokens.items():
            if isinstance(v, list):
                for i, item in enumerate(v):
                   self.int2token[item] = k+str(i)
            else:
                self.int2token[v] = k

    def _generate_all_hypotheses(self):
        """
        Generates all possible hypotheses.

        init:
        - all_hypotheses (list): List of all possible hypotheses.
        - num_all_h (int): Total number of possible hypotheses.
        """

        self.all_hypotheses = list(itertools.product(list(range(self.num_y)), repeat=self.num_x))
        self.num_all_h = len(self.all_hypotheses)  # Should be num_y**num_x
        if self.num_all_h != (self.num_y**self.num_x):
            raise Exception('wrong num_all_h')
    
    def _split_hypotheses_to_IO(self):
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        # Shuffle hypotheses indices
        shuffled_h_indices = list(range(self.num_all_h))
        random.shuffle(shuffled_h_indices)

        # Split hypotheses based on ratio
        self.num_I_h, self.num_O_h = self.num_IO_h

        I_h_indices = shuffled_h_indices[:self.num_I_h]
        O_h_indices = shuffled_h_indices[-self.num_O_h:]

        if self.num_I_h + self.num_O_h > len(shuffled_h_indices):
            raise Exception('not enough hypotheses in hypothesis universe')
        # Store the indices of hypotheses in training and testing sets
        self.I_h_indices = I_h_indices  # Indices into self.all_hypotheses
        self.O_h_indices = O_h_indices  # Indices into self.all_hypotheses

    def _construct_max_num_tables_and_shuffle(self):
        # Generate possible tables from respective hypothesesI_h_indices
        print(self.table_lengths)
        for length in self.table_lengths:
            num_total_I_tables = math.comb(len(self.I_h_indices), length)
            if num_total_I_tables <= self.max_num_tables:
                I_tables = list(itertools.combinations(self.I_h_indices, length))
            else:
                I_tables = sample_random_combinations(len(self.I_h_indices), length, self.max_num_tables)
                I_tables = [[self.I_h_indices[i] for i in train_possible_table] for train_possible_table in I_tables]
            random.shuffle(I_tables)
            self.I_tables[length] = I_tables

            num_total_O_tables = math.comb(len(self.O_h_indices), length)
            if num_total_O_tables <= self.max_num_tables:
                O_tables = list(itertools.combinations(self.O_h_indices, length))
            else:
                O_tables = sample_random_combinations(len(self.O_h_indices), length, self.max_num_tables)
                O_tables = [[self.O_h_indices[i] for i in test__possible_table] for test__possible_table in O_tables]
            random.shuffle(O_tables)
            self.O_tables[length] = O_tables

    # def _split_based_on_tables(self):
    #     random.seed(self.random_seed)
    #     np.random.seed(self.random_seed)

    #     for length in self.table_lengths:
    #         print(f'***** Table Length = {length} *****')
    #         #all_tables = list(itertools.combinations(range(self.num_all_h), length))
    #         #print(len(all_tables))
    #         num_total = math.comb(self.num_all_h, length)
    #         print(num_total, type(num_total))
    #         if num_total <= self.max_num_tables:
    #             time_A = time.time()
    #             possible_tables = list(itertools.combinations(range(self.num_all_h), length))
    #             time_B = time.time()
    #             print(f'generate time = {time_B-time_A}')
    #         elif num_total <= self.efficiency_threshold:
    #             time_A = time.time()
    #             print('apply reservoir sampling')
    #             possible_tables = []
    #             for i, combo in tqdm(enumerate(itertools.combinations(range(self.num_all_h), length))):
    #                 if i < self.max_num_tables:
    #                     # Initially fill up the reservoir
    #                     possible_tables.append(combo)
    #                 else:
    #                     # Once full, randomly replace elements with decreasing probability
    #                     r = random.randint(0, i)
    #                     if r < self.max_num_tables:
    #                         possible_tables[r] = combo
    #             print(len(possible_tables))
    #             time_B = time.time()
    #             print(f'generate time = {time_B-time_A}')
    #         else:
    #             time_A = time.time()
    #             print('apply efficient generation')
    #             possible_tables = sample_random_combinations(self.num_all_h, length, self.max_num_tables)
    #             print(len(possible_tables))
    #             time_B = time.time()
    #             print(f'generate time = {time_B-time_A}')


    #         random.shuffle(possible_tables)
    #         time_C = time.time()
    #         print(f'shuffle time = {time_C-time_B}')
    #         split_ratio = self.split_ratio
    #         if sum(split_ratio) != 1.0:
    #             total = sum(split_ratio)
    #             split_ratio = [r / total for r in split_ratio]

    #         total_tables = len(possible_tables)
    #         train_split = int(split_ratio[0] * total_tables)

    #         train_tables = possible_tables[:train_split]
    #         test__tables = possible_tables[train_split:]

    #         self.train_tables[length] = train_tables
    #         self.test__tables[length] = test__tables

    #     # All hypotheses are available in both splits
    #     self.train_h_indices = range(self.num_all_h)
    #     self.O_h_indices = range(self.num_all_h)

    def _sample_traintest_tables(self):
        train_info = self.train_info
        testI_info = self.testI_info
        testO_info = self.testO_info
        # check whether train test tables are enough
        lengths = list( set(train_info.keys()) | set(testI_info.keys()) | set(testO_info.keys()) )
        for length in lengths:
            num_requested = train_info.get(length, 0) + testI_info.get(length, 0)
            available_tables = self.I_tables[length]
            num_available = len(available_tables)
            if num_available < num_requested:
                raise Exception(f"Requested number of I tables ({num_requested}) "
                                    f"for length {length} exceeds available tables "
                                    f"({num_available}).")
            num_requested = testO_info.get(length, 0)
            available_tables = self.O_tables[length]
            num_available = len(available_tables)
            if num_available < num_requested:
                raise Exception(f"Requested number of O tables ({num_requested}) "
                                    f"for length {length} exceeds available tables "
                                    f"(num_available).")
        # sampling process
        for length in lengths:
            if train_info.get(length, 0) != 0:
                self.train_tables[length] = self.I_tables[length][:train_info.get(length, 0)]
            #print(train_info.get(length, 0))
            #print(testI_info.get(length, 0))
            self.testI_tables[length] = self.I_tables[length][-testI_info.get(length, 0):]
            #print(self.train_tables)
            #print(self.testI_tables)
            self.testO_tables[length] = self.O_tables[length][:testO_info.get(length, 0)]

    def _sample_train_tables(self):
        # Keep only lengths specified in train_info
        lengths_to_keep = set(self.train_info.keys())
        lengths_to_remove = set(self.train_tables.keys()) - lengths_to_keep
        for length in lengths_to_remove:
            del self.train_tables[length]

        # Sample train tables according to self.train_info
        for length, num_tables in self.train_info.items():
            if length in self.train_tables:
                available_tables = self.train_tables[length]
                if num_tables <= len(available_tables):
                    sampled_tables = random.sample(available_tables, num_tables)
                    self.train_tables[length] = sampled_tables
                else:
                    raise Exception(f"Requested number of training tables ({num_tables}) "
                                    f"for length {length} exceeds available tables "
                                    f"({len(available_tables)}).")
            else:
                raise Exception(f"No training tables of length {length} to sample from.")
        
    def _sample_test__tables(self):
        # Keep only lengths specified in test__info
        lengths_to_keep = set(self.test__info.keys())
        lengths_to_remove = set(self.test__tables.keys()) - lengths_to_keep
        for length in lengths_to_remove:
            del self.test__tables[length]

        # Sample test tables according to self.test__info
        for length, num_tables in self.test__info.items():
            if length in self.test__tables:
                available_tables = self.test__tables[length]
                if num_tables <= len(available_tables):
                    sampled_tables = random.sample(available_tables, num_tables)
                    self.test__tables[length] = sampled_tables
                else:
                    raise Exception(f"Requested number of testing tables ({num_tables}) "
                                    f"for length {length} exceeds available tables "
                                    f"({len(available_tables)}).")
            else:
                raise Exception(f"No testing tables of length {length} to sample from.")

    def _calculate_identifying_x(self, H):
        """
        Calculate the identifying_x_matrix (I) by finding the minimal set of positions
        needed to distinguish each hypothesis from all others.

        Parameters:
        - H (np.array): Matrix of hypotheses (rows: hypotheses, columns: positions).

        Returns:
        - I (np.array): Binary matrix where 1 indicates the position is necessary.
        """
        num_h, num_x = H.shape
        I = np.zeros_like(H, dtype=int)

        for i in range(num_h):
            D_ij_list = []
            for j in range(num_h):
                if j != i:
                    differing_positions = set()
                    for position in range(num_x):
                        if H[i, position] != H[j, position]:
                            differing_positions.add(position)
                    D_ij_list.append(differing_positions)

            found = False
            positions = list(range(num_x))
            for k in range(1, num_x+1):
                subsets = list(itertools.combinations(positions, k))
                random.shuffle(subsets)
                for S in subsets:
                    hits_all = all(any(pos in D_ij for pos in S) for D_ij in D_ij_list)
                    if hits_all:
                        for position in S:
                            I[i, position] = 1
                        found = True
                        break
                if found:
                    break
            if not found:
                raise Exception(f"No identifying set found for hypothesis {i}")

        return I

class DataloaderManager:
    def __init__(self, args, hmanager, n_steps, split, preshuffle, icl_sampling, iid_probability=None, icl_y_noise=None):
        self.h_prefix_format = args.h_prefix_format
        
        self.hmanager = hmanager
        self.split = split
        self.preshuffle = preshuffle
        self.dataset = HDataset(args=args, hmanager=self.hmanager, split=split, n_steps=n_steps)

        self.tokens = self.hmanager.tokens

        self.num_x = args.num_x
        self.max_table_length = args.max_table_length

        self.icl_k = args.icl_k
        self.icl_sampling = icl_sampling
        self.iid_probability = iid_probability
        self.icl_y_noise = icl_y_noise
        self.mix_prob_train1 = args.mix_prob_train1

        self.batch_size = args.batch_size
    def get_pytorch_dataloader(self):
        sampler = None
        shuffle = False

        # Define the collate function
        def collate_fn(batch):
            h_list = []
            i_list = []
            hH_idx_list = []
            H_list = []
            I_list = []
            xy_seq_list_info = {
                'xy_seq_list'      : [],
                'xy_seq_xmask_list': [],
                'xy_seq_ymask_list': [],
                'xy_seq_zmask_list': [],
                'xy_seq_hmask_list': [],
                'xy_seq_smask_list': [],
            }
            for h, i, H, I, hH_idx in batch:
                h_list.append(torch.tensor(h))
                i_list.append(torch.tensor(i))
                H_list.append(torch.tensor(H))
                I_list.append(torch.tensor(I))
                hH_idx_list.append(hH_idx)

                if self.icl_sampling in ['ordered', 'permutation', 'iid']:
                    xy_seq_info = generate_sequence_normal(h, self.icl_sampling)
                # elif self.icl_sampling == 'optimal':
                #     xy_seq, mask_sequence = generate_sequence_optimal(h, i, self.y_mask_value)
                elif self.icl_sampling == 'optimal':
                    xy_seq_info = generate_sequence_optimal(h, i)
                # elif self.icl_sampling == 'mix':
                #     xy_seq, mask_sequence = generate_sequence_mix(h, i, self.y_mask_value)
                else:
                    raise ValueError("Invalid dataloader_type.")
                #print(xy_seq)

                for key in xy_seq_list_info.keys():
                    xy_seq_list_info[key].append(xy_seq_info[key[:-5]])
               
            # Convert H_list and I_list to lists of tensors
            H_list = [H.clone().detach().long() for H in H_list]
            
            spH_list = []
            z_list = []  # h_index_in_spH_list
            for H, hH_idx in zip(H_list, hH_idx_list):
                spH, Z = construct_shuffled_nanned_table(self.max_table_length, H, self.tokens['nan'], self.tokens['hH_Z'], preshuffle=self.preshuffle)
                spH_list.append(spH)
                z_list.append(Z[hH_idx])

            z_suffix_list       = [torch.tensor([self.tokens['>']]*1 + [z  ], dtype=torch.long) for z in z_list]
            z_suffix_xmask_list = [torch.tensor([0.0             ]*1 + [0.0], dtype=torch.long) for z in z_list]
            z_suffix_ymask_list = [torch.tensor([0.0             ]*1 + [0.0], dtype=torch.long) for z in z_list]
            z_suffix_zmask_list = [torch.tensor([0.0             ]*1 + [1.0], dtype=torch.long) for z in z_list]
            z_suffix_hmask_list = [torch.tensor([0.0             ]*1 + [0.0], dtype=torch.long) for z in z_list]
            z_suffix_smask_list = [torch.tensor([1.0             ]*1 + [0.0], dtype=torch.long) for z in z_list]
            z_suffix_list_info ={
                'z_suffix_list'      : z_suffix_list,
                'z_suffix_xmask_list': z_suffix_xmask_list,
                'z_suffix_ymask_list': z_suffix_ymask_list,
                'z_suffix_zmask_list': z_suffix_zmask_list,
                'z_suffix_hmask_list': z_suffix_hmask_list,
                'z_suffix_smask_list': z_suffix_smask_list,
            }

            I_list = [I.clone().detach().long() for I in I_list]

            #H_prefix, H_prefix_mask = self.generate_H_prefix(h_matrices_tensor)
            spH_prefix_list_info = self.generate_spH_prefix(spH_list)
            #return (xy_list, mask_list, spH_prefix_list, spH_prefix_mask_list,
            #        h_list, i_list, hH_idx_list, y_list, y_suffix_list, y_suffix_mask_list, spH_list, H_list, I_list)
            return {'xy_seq_list_info'    : xy_seq_list_info,
                    'spH_prefix_list_info': spH_prefix_list_info,
                    'z_suffix_list_info'  : z_suffix_list_info,
                    'h_list': h_list,
                    'i_list': i_list,
                    'hH_idx_list': hH_idx_list,
                    'H_list': H_list,
                    'I_list': I_list,
                    'spH_list': spH_list,
                    'z_list': z_list,
                    }

        # Helper functions for generating sequences
        def generate_sequence_normal(h, icl_sampling): # output dictionary of tensor
            x_seq = []
            y_seq = []
            if icl_sampling == 'ordered':
                position_indices = np.arange(self.num_x)
            elif icl_sampling == 'permutation':
                position_indices = np.random.permutation(np.arange(self.num_x))
            elif icl_sampling == 'iid':
                if self.iid_probability is None:
                    position_indices = np.random.choice(self.num_x, size=self.icl_k, replace=True)
                else:
                    position_indices = np.random.choice(self.num_x, size=self.icl_k, replace=True, p=self.iid_probability)
            else:
                raise Exception(f'wrong if icl_sampling == {icl_sampling}')
            # print('position_indices =', position_indices)
            for position_index in position_indices:
                x = self.tokens['xs'][position_index]
                y = self.tokens['ys'][h[position_index]]
                x_seq.append(x)
                y_seq.append(y)

            # Interleave x_seq and y_seq to create xy_seq
            xy_seq = []
            xy_seq_xmask = []
            xy_seq_ymask = []
            xy_seq_zmask = []
            xy_seq_hmask = []
            xy_seq_smask = []
            for x, y in zip(x_seq, y_seq):
                if random.random() < self.icl_y_noise:
                    #print(y)
                    #print(self.tokens['ys'])
                    #print(y)
                    n_y_pool = [candidate for candidate in self.tokens['ys'] if candidate != y]
                    #print(n_y_pool)
                    n_y = np.random.choice(n_y_pool)
                    #print(n_y)
                    xy_seq  .extend([x  , n_y, self.tokens[',']])
                else:
                    xy_seq  .extend([x  , y  , self.tokens[',']])
                xy_seq_xmask.extend([1.0, 0.0, 0.0             ])
                xy_seq_ymask.extend([0.0, 1.0, 0.0             ])
                xy_seq_zmask.extend([0.0, 0.0, 0.0             ])
                xy_seq_hmask.extend([0.0, 0.0, 0.0             ])
                xy_seq_smask.extend([0.0, 0.0, 1.0             ])
            
            xy_seq_info = {
                'xy_seq'      : torch.tensor(xy_seq      [:-1], dtype=torch.long),
                'xy_seq_xmask': torch.tensor(xy_seq_xmask[:-1], dtype=torch.long),
                'xy_seq_ymask': torch.tensor(xy_seq_ymask[:-1], dtype=torch.long),
                'xy_seq_zmask': torch.tensor(xy_seq_zmask[:-1], dtype=torch.long),
                'xy_seq_hmask': torch.tensor(xy_seq_hmask[:-1], dtype=torch.long),
                'xy_seq_smask': torch.tensor(xy_seq_smask[:-1], dtype=torch.long),
            }
            return xy_seq_info

        def generate_sequence_optimal(h, i):
            x_seq = []
            y_seq = []
            position_indices = np.where(i == 1)[0]
            np.random.shuffle(position_indices)
            for position_index in position_indices:
                x = self.tokens['xs'][position_index]
                y = self.tokens['ys'][h[position_index]]
                x_seq.append(x)
                y_seq.append(y)

            x_seq = repeat_list_to_length(x_seq, self.icl_k)
            y_seq = repeat_list_to_length(y_seq, self.icl_k)

            indices = list(range(self.icl_k))
            random.shuffle(indices)
    
            # Apply the same permutation to all lists
            x_seq = [x_seq[i] for i in indices]
            y_seq = [y_seq[i] for i in indices]

            # Interleave x_seq and y_seq to create xy_seq
            xy_seq = []
            xy_seq_xmask = []
            xy_seq_ymask = []
            xy_seq_zmask = []
            xy_seq_hmask = []
            xy_seq_smask = []
            for x, y in zip(x_seq, y_seq):
                xy_seq      .extend([x  , y  , self.tokens[',']])
                xy_seq_xmask.extend([1.0, 0.0, 0.0             ])
                xy_seq_ymask.extend([0.0, 1.0, 0.0             ])
                xy_seq_zmask.extend([0.0, 0.0, 0.0             ])
                xy_seq_hmask.extend([0.0, 0.0, 0.0             ])
                xy_seq_smask.extend([0.0, 0.0, 1.0             ])

            xy_seq_info = {
                'xy_seq'      : torch.tensor(xy_seq      [:-1], dtype=torch.long),
                'xy_seq_xmask': torch.tensor(xy_seq_xmask[:-1], dtype=torch.long),
                'xy_seq_ymask': torch.tensor(xy_seq_ymask[:-1], dtype=torch.long),
                'xy_seq_zmask': torch.tensor(xy_seq_zmask[:-1], dtype=torch.long),
                'xy_seq_hmask': torch.tensor(xy_seq_hmask[:-1], dtype=torch.long),
                'xy_seq_smask': torch.tensor(xy_seq_smask[:-1], dtype=torch.long),
            }
            return xy_seq_info

        # def generate_sequence_mix(h, i, y_mask_value):
        #     is_normal = np.random.rand() < mix_prob_train1
        #     if is_normal:
        #         return generate_sequence_normal(h, y_mask_value)
        #     else:
        #         return generate_sequence_optimal(h, i, y_mask_value)

        # def generate_sequences_test(h, i):
        #     sequences = []
        #     located_position_indices = np.where(i == 1)[0]
        #     located_xs = []
        #     located_ys = []
        #     y_mask_sequence_base = []
        #     for position_index in located_position_indices:
        #         y = h[position_index]
        #         x = position_index
        #         located_xs.append(x)
        #         located_ys.append(y)
        #         y_mask_sequence_base.append(0.0)  # Mask 0.0 for located y

        #     # Interleave located_xs and located_ys
        #     located_xy_seq = []
        #     mask_sequence_base = []
        #     for x, y, mask_y in zip(located_xs, located_ys, y_mask_sequence_base):
        #         located_xy_seq.extend([x, y])
        #         mask_sequence_base.extend([0.0, mask_y])

        #     # Repeat prefix if needed
        #     if prefix_repeat is not None:
        #         located_xy_seq = located_xy_seq * prefix_repeat
        #         mask_sequence_base = mask_sequence_base * prefix_repeat

        #     # For each additional position, create a sample
        #     additional_position_indices = np.arange(self.n)
        #     for position_index in additional_position_indices:
        #         x_seq = located_xy_seq.copy()
        #         mask_seq = mask_sequence_base.copy()
        #         y = h[position_index]
        #         x = position_index
        #         x_seq.extend([x, y])
        #         mask_seq.extend([0.0, 1])  # Mask 0.0 for x, 1 for y
        #         sequences.append((x_seq, mask_seq))
        #     return sequences

        # Create the DataLoader
        data_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=0
        )

        return data_loader

    def generate_H_prefix(self, H_list):
        """
        Converts a list of H into a sequence H_prefix.

        Parameters:
        - H_list: List of tensors, each of shape (m_i, n) containing the hypotheses.

        Returns:
        - H_prefix: Tensor of shape (batch_size, sequence_length) containing the interleaved x, y sequences.
        - H_prefix_mask: Tensor of shape (batch_size, sequence_length) containing all zeros.
        """
        batch_size = len(H_list)
        H_prefix_list = []

        for i in range(batch_size):
            H_prefix_seq = []
            H = H_list[i]
            m_i, n = H.shape

            for h in H:
                # Generate interleaved x, y sequence for this hypothesis
                x_seq = [self.tokens['xs'][x_index] for x_index in range(n)]
                y_seq = [self.tokens['ys'][y_index] for y_index in h.tolist()]
                xy_seq = []
                for x, y in zip(x_seq, y_seq):
                    xy_seq.extend([x, y])
                # Append the xy_seq to H_prefix_seq
                H_prefix_seq.extend(xy_seq)
                # Add a pad token to separate hypotheses
                H_prefix_seq.append(self.pad_token)
            # Remove the last pad token and add three pad tokens at the end
            if H_prefix_seq:
                H_prefix_seq = H_prefix_seq[:-1] + [self.pad_token] * 3
            else:
                H_prefix_seq = [self.pad_token] * 3
            H_prefix_list.append(H_prefix_seq)

        # Pad the sequences to the maximum length
        # max_length = max(len(seq) for seq in H_prefix_list)
        # H_prefix_padded = []
        # for seq in H_prefix_list:
        #     pad_len = max_length - len(seq)
        #     seq_padded = seq + [pad_token] * pad_len
        #     H_prefix_padded.append(seq_padded)
        # H_prefix = torch.tensor(H_prefix_padded, dtype=torch.long)
        # H_prefix_mask = torch.zeros_like(H_prefix, dtype=torch.long)
        H_prefix_list = [torch.tensor(H_prefix) for H_prefix in H_prefix_list]
        H_prefix_mask_list = [torch.zeros_like(H_prefix) for H_prefix in H_prefix_list]
        return H_prefix_list, H_prefix_mask_list

    def generate_spH_prefix(self, spH_list):
        """
        Converts a list of h_matrices into a sequence H_prefix.

        Parameters:
        - h_matrices_tensor: List of tensors, each of shape (m, n) containing the hypotheses.

        Returns:
        - H_prefix: Tensor of shape (batch_size, sequence_length) containing the interleaved x, y sequences.
        - H_prefix_mask: Tensor of shape (batch_size, sequence_length) containing all zeros.
        """
        batch_size = len(spH_list)
        spH_prefix_list = []
        spH_prefix_xmask_list = []
        spH_prefix_ymask_list = []
        spH_prefix_zmask_list = []
        spH_prefix_hmask_list = []
        spH_prefix_smask_list = []

        for spH in spH_list:
            spH_prefix       = []
            spH_prefix_xmask = []
            spH_prefix_ymask = []
            spH_prefix_zmask = []
            spH_prefix_hmask = []
            spH_prefix_smask = []
            m, n = spH.shape
            
            for h in spH:
                # Add a pad token to separate hypotheses
                spH_prefix      .extend([self.tokens['pad']] * 2)
                spH_prefix_xmask.extend([0.0               ] * 2)
                spH_prefix_ymask.extend([0.0               ] * 2)
                spH_prefix_zmask.extend([0.0               ] * 2)
                spH_prefix_hmask.extend([0.0               ] * 2)
                spH_prefix_smask.extend([1.0               ] * 2)
                if self.h_prefix_format == 1:
                    spH_prefix      .extend([h[-1]] + [self.tokens['>']] * 1)
                    spH_prefix_xmask.extend([0.0  ] + [0.0             ] * 1)
                    spH_prefix_ymask.extend([0.0  ] + [0.0             ] * 1)
                    spH_prefix_zmask.extend([0.0  ] + [0.0             ] * 1)
                    spH_prefix_hmask.extend([1.0  ] + [0.0             ] * 1)
                    spH_prefix_smask.extend([0.0  ] + [1.0             ] * 1)
                # Generate interleaved x, y sequence for this hypothesis
                if h[0] != self.tokens['nan']:
                    x_seq = [self.tokens['xs'][x_index] for x_index in range(n-1)]
                    y_seq = [self.tokens['ys'][y_index] for y_index in h[:-1].tolist()]
                else:
                    x_seq = [self.tokens['nan'] for x_index in range(n-1)]
                    y_seq = [self.tokens['nan'] for y_index in h[:-1].tolist()]
                xy_seq       = []
                xy_seq_xmask = []
                xy_seq_ymask = []
                xy_seq_zmask = []
                xy_seq_hmask = []
                xy_seq_smask = []
                for x, y in zip(x_seq, y_seq):
                    xy_seq      .extend([x  , y  , self.tokens[',']])
                    xy_seq_xmask.extend([1.0, 0.0, 0.0             ])
                    xy_seq_ymask.extend([0.0, 1.0, 0.0             ])
                    xy_seq_zmask.extend([0.0, 0.0, 0.0             ])
                    xy_seq_hmask.extend([0.0, 0.0, 0.0             ])
                    xy_seq_smask.extend([0.0, 0.0, 1.0             ])
                # Append the xy_seq to H_prefix_seq
                spH_prefix      .extend(xy_seq      [:-1])
                spH_prefix_xmask.extend(xy_seq_xmask[:-1])
                spH_prefix_ymask.extend(xy_seq_ymask[:-1])
                spH_prefix_zmask.extend(xy_seq_zmask[:-1])
                spH_prefix_hmask.extend(xy_seq_hmask[:-1])
                spH_prefix_smask.extend(xy_seq_smask[:-1])
                # add index to the prefix
                if self.h_prefix_format == 0:
                    spH_prefix      .extend([self.tokens['>']] * 1 + [h[-1]])
                    spH_prefix_xmask.extend([0.0             ] * 1 + [0.0  ])
                    spH_prefix_ymask.extend([0.0             ] * 1 + [0.0  ])
                    spH_prefix_zmask.extend([0.0             ] * 1 + [0.0  ])
                    spH_prefix_hmask.extend([0.0             ] * 1 + [1.0  ])
                    spH_prefix_smask.extend([1.0             ] * 1 + [0.0  ])
            
            spH_prefix = spH_prefix + [self.tokens['pad']] * 4
            spH_prefix_xmask.extend(  [0.0               ] * 4)
            spH_prefix_ymask.extend(  [0.0               ] * 4)
            spH_prefix_zmask.extend(  [0.0               ] * 4)
            spH_prefix_hmask.extend(  [0.0               ] * 4)
            spH_prefix_smask.extend(  [1.0               ] * 4)
            
            spH_prefix_list      .append(spH_prefix      )
            spH_prefix_xmask_list.append(spH_prefix_xmask)
            spH_prefix_ymask_list.append(spH_prefix_ymask)
            spH_prefix_zmask_list.append(spH_prefix_zmask)
            spH_prefix_hmask_list.append(spH_prefix_hmask)
            spH_prefix_smask_list.append(spH_prefix_smask)

        spH_prefix_list       = [torch.tensor(item, dtype=torch.long) for item in spH_prefix_list      ]
        spH_prefix_xmask_list = [torch.tensor(item, dtype=torch.long) for item in spH_prefix_xmask_list]
        spH_prefix_ymask_list = [torch.tensor(item, dtype=torch.long) for item in spH_prefix_ymask_list]
        spH_prefix_zmask_list = [torch.tensor(item, dtype=torch.long) for item in spH_prefix_zmask_list]
        spH_prefix_hmask_list = [torch.tensor(item, dtype=torch.long) for item in spH_prefix_hmask_list]
        spH_prefix_smask_list = [torch.tensor(item, dtype=torch.long) for item in spH_prefix_smask_list]

        spH_prefix_list_info = {
            'spH_prefix_list'      : spH_prefix_list      ,
            'spH_prefix_xmask_list': spH_prefix_xmask_list,
            'spH_prefix_ymask_list': spH_prefix_ymask_list,
            'spH_prefix_zmask_list': spH_prefix_zmask_list,
            'spH_prefix_hmask_list': spH_prefix_hmask_list,
            'spH_prefix_smask_list': spH_prefix_smask_list,
        }
        return spH_prefix_list_info
    
class HDataset(Dataset):
    def __init__(self, args, hmanager, split, n_steps):
        self.hmanager = hmanager
        self.split = split

        self.icl_sampling = args.icl_sampling
        self.mix_prob_train1 = args.mix_prob_train1
        self.batch_size = args.batch_size
        self.n_steps = n_steps

        self.all_hypotheses = hmanager.all_hypotheses  # Access to all hypotheses

        if self.split == 'train':
            #print('hmanager.train_tables.keys()')
            #print(hmanager.train_tables.keys())
            self.tables = hmanager.train_tables
            self.hypotheses_indices = hmanager.I_h_indices
        elif self.split == 'testI':
            self.tables = hmanager.testI_tables
            self.hypotheses_indices = hmanager.I_h_indices
        elif self.split == 'testO':
            self.tables = hmanager.testO_tables
            self.hypotheses_indices = hmanager.O_h_indices
        else:
            raise ValueError("Invalid split. Must be 'train' or 'test_'.")

        # Prepare a list of lengths for sampling
        self.table_lengths = list(self.tables.keys())

        if self.icl_sampling == 'test':
            # For test, create a list of all possible combinations
            self.all_test__items = []
            for length in self.table_lengths:
                for hA_idx_list in self.tables[length]: # Indices into all_hypotheses
                    H = np.array([self.all_hypotheses[hA_idx] for hA_idx in hA_idx_list])
                    I = self.hmanager._calculate_identifying_x(H)
                    for hH_idx in range(len(h_indices)):
                        self.all_test__items.append((H, I, hH_idx))
            self.length = len(self.all_test__items)
        else:
            # For train, set length to n_steps * batch_size
            self.length = self.n_steps * self.batch_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.icl_sampling == 'test':
            # Get the item from self.all_test__items
            H, I, hH_idx = self.all_test__items[idx]
            h = H[hH_idx]
            i = I[hH_idx]
            return h, i, H, I, hH_idx
        else:
            # Randomly sample a length
            length = random.choice(self.table_lengths)
            #print(length)
            # Randomly sample a table
            #print(self.tables[9][:5])
            #print(type(self.tables))
            #print(self.tables.keys())
            #print(len(self.tables[length]))
            hA_idx_list = random.choice(self.tables[length])
            
            H = np.array([self.all_hypotheses[hA_idx] for hA_idx in hA_idx_list])
            I = self.hmanager._calculate_identifying_x(H)
            # Randomly select a hypothesis from the table
            hH_idx = random.randint(0, length - 1)
            
            h = H[hH_idx]
            i = I[hH_idx]
            return h, i, H, I, hH_idx



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch In-context Learning Training Code')
    parser.set_defaults(augment=True)
    args = parser.parse_args()

    args.random_seed = 2023

    ### hmanager
    args.num_x = 4
    args.num_y = 2
    args.max_table_length = 4
    # table_lengths
    args.split_based_on = 'table'
    # split_ratio
    # train_info
    # test__info

    ### dataloader
    split = 'train'

    args.icl_k = 4
    args.icl_sampling = 'ordered'
    args.sampling_disparity = 1.0
    args.icl_y_noise = 1.0
    args.h_prefix_format = 0
    args.mix_prob_train1 = 0.5  # Probability for 'mix' mode

    args.batch_size = 2  # Number of samples per batch
    args.n_steps = 32  # Number of steps per epoch


    # Parameters
    mode = 'binary'
    if args.split_based_on == 'hypothesis':
        n = 4  # Number of elements in the mode
        random_seed = 2023
        k = 4  # Number of x-y pairs per sequence
        batch_size = 2  # Number of samples per batch
        n_steps = 32  # Number of steps per epoch
        mix_prob_train1 = 0.5  # Probability for 'mix' mode
        table_lengths = [4]
        split_ratio = [0.5, 0.5]  # Ratios for train and test splits
        train_info = {4: 1820}  # Number of train tables to sample per length
        test__info = {4: 1820}  # Number of test tables to sample per length
        # n = 4  # Number of elements in the mode
        # random_seed = 2024
        # k = 3  # Number of x-y pairs per sequence
        # batch_size = 2  # Number of samples per batch
        # n_steps = 32  # Number of steps per epoch
        # mix_prob_train1 = 0.5  # Probability for 'mix' mode
        # table_lengths = [3, 4, 5, 6, 7, 8]
        # split_ratio = [0.5, 0.5]  # Ratios for train and test splits
        # train_info = {3: 500, 4: 1000, 5: 2000, 6: 4000, 7: 8000, 8: 8000}  # Number of train tables to sample per length
        # test__info = {3: 500, 4: 500, 5: 1000, 6: 1000, 7: 1000, 8: 1000}  # Number of test tables to sample per length
    if args.split_based_on == 'table':
        if args.num_x == 2:
            table_lengths = [2]
            split_ratio = [1, 0]  # Ratios for train and test splits
            train_info = {2: 6}  # Number of train tables to sample per length
            test__info = {2: 0}  # Number of test tables to sample per length
        if args.num_x == 4:
            table_lengths = [4]#, 5, 6, 7]
            split_ratio = [0.7, 0.3]
            train_info = {4: 1274}#, 5: 1274, 6: 1274, 7:1274}  # Number of train tables to sample per length
            test__info = {4: 546}#, 5: 546, 6: 546, 7:546}  # Number of train tables to sample per length
        if args.num_x == 5:
            table_lengths = [3, 4, 5, 6, 7]
            split_ratio = [2/3, 1/3]
            train_info = {3: 3000, 4: 3000, 5: 3000, 6: 3000, 7: 3000}  # Number of train tables to sample per length
            test__info = {3:1500, 4: 1500, 5: 1500, 6: 1500, 7: 1500}  # Number of train tables to sample per length

    # Initialize the HypothesisManager
    hmanager = HypothesisManager(
        args,
        table_lengths=table_lengths,
        split_ratio=split_ratio,
        train_info=train_info,
        test__info=test__info
    )
    
    dmanager = DataloaderManager(
        args,
        hmanager = hmanager,
        split = split,
        preshuffle = False,
        icl_sampling = 'iid',
        iid_probability = None,
        icl_y_noise = args.icl_y_noise
    )

    # Get the data loader for testing
    dataloader = dmanager.get_pytorch_dataloader()

    # Iterate through the dataloader
    if 1:
        count = 1
        for batch in dataloader:

            print("-" * 10, count, "-" * 10)
            # Unpack the batch

            print("H_list:")
            print(batch['H_list'])
            print("I_list:")
            print(batch['I_list'])
            print("hH_idx_list")
            print(batch['hH_idx_list'])
            print("spH_list")
            print(batch['spH_list'])
            print("z_list:")
            print(batch['z_list'])
            print("h_list:")
            print(batch['h_list'])
            print("i_list:")
            print(batch['i_list'])

            print("spH_prefix_list_info:")
            print(batch['spH_prefix_list_info']['spH_prefix_list'])
            print(batch['spH_prefix_list_info']['spH_prefix_xmask_list'])
            print(batch['spH_prefix_list_info']['spH_prefix_ymask_list'])
            print(batch['spH_prefix_list_info']['spH_prefix_zmask_list'])
            print(batch['spH_prefix_list_info']['spH_prefix_hmask_list'])
            print(batch['spH_prefix_list_info']['spH_prefix_smask_list'])

            print("xy_seq_list_info:")
            print(batch['xy_seq_list_info']['xy_seq_list'])
            print(batch['xy_seq_list_info']['xy_seq_xmask_list'])
            print(batch['xy_seq_list_info']['xy_seq_ymask_list'])
            print(batch['xy_seq_list_info']['xy_seq_zmask_list'])
            print(batch['xy_seq_list_info']['xy_seq_hmask_list'])
            print(batch['xy_seq_list_info']['xy_seq_smask_list'])
            
            print("z_suffix_list_info:")
            print(batch['z_suffix_list_info']['z_suffix_list'])
            print(batch['z_suffix_list_info']['z_suffix_xmask_list'])
            print(batch['z_suffix_list_info']['z_suffix_ymask_list'])
            print(batch['z_suffix_list_info']['z_suffix_zmask_list'])
            print(batch['z_suffix_list_info']['z_suffix_hmask_list'])
            print(batch['z_suffix_list_info']['z_suffix_smask_list'])
            if count == 1:
                break  # Remove this line to iterate over the entire dataset