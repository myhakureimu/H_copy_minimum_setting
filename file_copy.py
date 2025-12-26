import shutil

# Source file
prefix = 'EXP_H_noise/IO_16'
source_file = f'{prefix}.py'

# Copy the file to run1.py through run8.py, overwriting if necessary
for i in ['A','B','C','D']:
    target_file = f'{prefix}{i}.py'
    shutil.copy(source_file, target_file)
    print(f"Copied {source_file} to {target_file}, overwriting if it existed")