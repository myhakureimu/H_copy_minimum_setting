import os

for i in range(0,4):
    old_filename = f"EXP_FourGeneralization/IO_{i}.py"
    new_filename = f"EXP_FourGeneralization/IOS_{i}.py"
    
    # Read contents of A{i}.py
    with open(old_filename, 'r') as f:
        content = f.read()
    
    # Replace the line gpuIdxStr = '1' with gpuIdxStr = '2'
    new_content = content.replace(
        "HEAD, exp_name = 'ICL', 'IOHypothesis'",
        "HEAD, exp_name = 'ICL', 'IOHypothesis+Size'"
    )
    
    # Write to B{i}.py
    with open(new_filename, 'w') as f:
        f.write(new_content)
    
    print(f"Copied {old_filename} to {new_filename} with gpuIdxStr updated.")