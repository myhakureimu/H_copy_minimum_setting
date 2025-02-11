import os

for i in range(0,8):
    old_filename = f"EXP_NUMTRAIN/IO_t_A{i}.py"
    new_filename = f"EXP_NUMTRAIN/IO_t_B{i}.py"
    
    # Read contents of A{i}.py
    with open(old_filename, 'r') as f:
        content = f.read()
    
    # Replace the line gpuIdxStr = '1' with gpuIdxStr = '2'
    new_content = content.replace("gpuIdxStr = '0'", "gpuIdxStr = '1'")
    
    # Write to B{i}.py
    with open(new_filename, 'w') as f:
        f.write(new_content)
    
    print(f"Copied {old_filename} to {new_filename} with gpuIdxStr updated.")