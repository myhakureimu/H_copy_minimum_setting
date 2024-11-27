import torch

def construct_shuffled_nanned_table(max_table_length, table, padded_token, table_index_tokens, preshuffle = False):
    """
    Constructs a shuffled and padded table with index tokens.

    Parameters:
    -----------
    max_table_length : int
        The maximum length of the padded table, i.e., the number of rows in the output table.
    
    table : torch.Tensor or list of lists
        A tensor (or list of lists) representing the original table of size (N, F), 
        where N is the number of rows and F is the number of features per row.

    padded_token : int or float
        The value used to pad the table if the number of rows in the table is less than max_table_length.

    table_index_tokens : list of int
        A list of index tokens of size max_table_length, which will be shuffled and added as the first column of the table.
        The first `N` elements in the shuffled index correspond to the original rows' indices after shuffling.

    Returns:
    --------
    shuffled_padded_table : list of lists
        A shuffled and padded table of size (max_table_length, F+1), where the first column contains the shuffled index tokens
        and the remaining columns contain the original table and padded rows.
    
    original_indices_in_shuffled : list of int
        A list of indices indicating where the original rows of the table appear in the shuffled and padded table.
        These indices refer to the original rows' new positions after shuffling.
    """

    # Convert the input table to a PyTorch tensor if it isn't already
    #table = torch.tensor(table)

    # Determine the number of padded rows needed
    current_length = table.shape[0]
    padding_needed = max_table_length - current_length

    # (1) Create the padding rows filled with the padded_token
    padded_rows = torch.full((padding_needed, table.shape[1]), padded_token)

    # Concatenate the original table with the padded rows => padded_table
    padded_table = torch.cat((table, padded_rows), dim=0)

    # (2) Shuffle the table_index_tokens to get shuffle_table_index_tokens
    if preshuffle:
        shuffled_table_index_tokens = torch.tensor(table_index_tokens)[torch.randperm(max_table_length)]
    else:
        shuffled_table_index_tokens = torch.tensor(table_index_tokens)

    # (3) First len(table) elements are the index of the original rows in the shuffled order
    original_indices_in_shuffled = shuffled_table_index_tokens[:current_length].tolist()

    # (4) Concatenate the padded index tokens as the first column of padded_table
    padded_table_with_indices = torch.cat(
        (padded_table,shuffled_table_index_tokens.unsqueeze(1)), dim=1
    )

    # (5) Shuffle the padded_table along dimension 0 (shuffle rows)
    shuffled_padded_table = padded_table_with_indices[torch.randperm(max_table_length)]
    #shuffled_padded_table = padded_table_with_indices

    # (6) Return the shuffled_padded_table and the original indices in shuffled order
    return shuffled_padded_table, original_indices_in_shuffled

# Test the function
if __name__ == "__main__":
    table = torch.tensor([[1, 0], [1, 1]])  # Input table
    padded_token = 9  # Padded token
    max_table_length = 4  # Maximum table length
    table_index_tokens = [7, 4, 6, 5]  # Index tokens

    shuffled_table, original_indices = construct_shuffled_nanned_table(max_table_length, table, padded_token, table_index_tokens)

    print("Shuffled padded table with index tokens:")
    print(shuffled_table)

    print("\nOriginal indices in shuffled table:", original_indices)
