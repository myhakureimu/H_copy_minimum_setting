import torch

# def split_tensor_by_value(tensor, pad_token=6):
#     # Identify indices where the pad_token appears
#     pad_indices = torch.where(tensor == pad_token)[0]
    
#     # Add the end of the tensor as the last split point
#     pad_indices = torch.cat((pad_indices, torch.tensor([len(tensor)])))
    
#     # Split the tensor into sub-tensors
#     sub_tensors = []
#     for i in range(len(pad_indices) - 1):
#         start = pad_indices[i] + 1
#         end = pad_indices[i + 1]
#         if start < end:  # Ensure there is data to include
#             sub_tensors.append(tensor[start:end])
    
#     return sub_tensors

def split_tensor_by_value(tensor, pad_token=6):
    # Identify indices where the pad_token appears
    pad_indices = torch.where(tensor == pad_token)[0]
    
    # Add the start and end of the tensor as split points
    split_points = torch.cat((torch.tensor([-1]), pad_indices, torch.tensor([len(tensor)])))
    
    # Split the tensor into sub-tensors
    sub_tensors = []
    for i in range(len(split_points) - 1):
        start = split_points[i] + 1
        end = split_points[i + 1]
        if start < end:  # Ensure there is data to include
            sub_tensors.append(tensor[start:end])
    
    return sub_tensors

def hypothesis_seq_to_z_x2y(hypothesis_seq, predict_token=8, comma_token=9):
    #print('hypothesis_seq')
    #print(hypothesis_seq)
    sub_tensors = split_tensor_by_value(hypothesis_seq, predict_token)
    #print(sub_tensors)
    xy_seq, z_seq = sub_tensors[0], sub_tensors[1]
    xy_list = split_tensor_by_value(xy_seq, comma_token)
    x2y = {}
    for xy in xy_list:
        x2y[xy[0].item()] = xy[1].item()
    return z_seq.item(), x2y

def spH_prefix_to_z2x2y(spH_prefix, pad_token=6, predict_token=8, comma_token=9):
    hypothesis_seq_list = split_tensor_by_value(spH_prefix, pad_token)
    #print('hypothesis_seq_list')
    #print(hypothesis_seq_list)
    z2x2y = {}
    for hypothesis_seq in hypothesis_seq_list:
        z, x2y = hypothesis_seq_to_z_x2y(hypothesis_seq, predict_token, comma_token)
        z2x2y[z] = x2y
    return z2x2y

def xy_seq_to_x2y(xy_seq, comma_token=9):
    #print('xy_seq')
    #print(xy_seq)
    xy_list = split_tensor_by_value(xy_seq, comma_token)
    #print('xy_list')
    #print(xy_list)
    x2y = {}
    for xy in xy_list:
        x2y[xy[0].item()] = xy[1].item()
    return x2y

def find_valid_zs(spH_prefix, xy_seq, pad_token=6, predict_token=8, comma_token=9):
    #print('spH_prefix')
    #print(spH_prefix)
    z2x2y = spH_prefix_to_z2x2y(spH_prefix, pad_token, predict_token, comma_token)
    x2y = xy_seq_to_x2y(xy_seq, comma_token=comma_token)
    valid_zs = []
    for z in z2x2y.keys():
        valid = True
        for x in x2y.keys():
            if (x not in z2x2y[z].keys()) or (x2y[x] != z2x2y[z][x]):
                valid = False
                break
        if valid:
            valid_zs.append(z)
    return valid_zs



if __name__ == "__main__":
    # Example usage
    tensor = torch.tensor([6, 6, 0, 5, 9, 1, 5, 9, 2, 5, 9, 3, 4, 8, 13, 6, 6, 0,
                        5, 9, 1, 4, 9, 2, 4, 9, 3, 5, 8, 10, 6, 6, 0, 4, 9, 1,
                        4, 9, 2, 4, 9, 3, 5, 8, 12, 6, 6, 0, 4, 9, 1, 5, 9, 2,
                        5, 9, 3, 4, 8, 11, 6, 6, 6, 6])

    z2dictionary = spH_prefix_to_dictionary(tensor)
    print(z2dictionary)