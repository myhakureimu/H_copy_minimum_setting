import shutil

# Source file
prefix = 'YNOISE/TableGeneralization'
source_file = f'{prefix}0.py'

# Copy the file to run1.py through run8.py, overwriting if necessary
for i in range(1, 6):
    target_file = f'{prefix}{i}.py'
    shutil.copy(source_file, target_file)
    print(f"Copied {source_file} to {target_file}, overwriting if it existed")