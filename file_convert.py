import os

for i in range(0,4):
    old_filename = f"EXP_Diversity/IO_1h16_{i}.py"
    new_filename = f"EXP_Diversity/IO_1h24_{i}.py"
    
    # Read contents of A{i}.py
    with open(old_filename, 'r') as f:
        content = f.read()
    
    # Replace the line gpuIdxStr = '1' with gpuIdxStr = '2'
    new_content = content.replace(
        "num_training_hypotheses, num_training_tables = 16, 0",
        "num_training_hypotheses, num_training_tables = 24, 0"
    )
    
    # Write to B{i}.py
    with open(new_filename, 'w') as f:
        f.write(new_content)
    
    print(f"Copied {old_filename} to {new_filename} with gpuIdxStr updated.")