import shutil

# Source file
source_file = "run0.py"

# Copy the file to run1.py through run8.py, overwriting if necessary
for i in range(1, 8):
    target_file = f"run{i}.py"
    shutil.copy(source_file, target_file)
    print(f"Copied {source_file} to {target_file}, overwriting if it existed")