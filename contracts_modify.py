import os
import sys
import platform 

env_path = sys.prefix

if platform.system() == 'Windows':
    file_path = os.path.join(env_path, 'lib', 'site-packages', 'contracts', 'library', 'array_ops.py')
else:
    file_path = os.path.join(env_path, 'lib', 'python3.8', 'site-packages', 'contracts', 'library', 'array_ops.py')

with open(file_path, 'r') as file:
    lines = file.readlines()

lines = [line.replace('np.int,', 'int,') for line in lines]
lines = [line.replace('np.float,', 'float,') for line in lines]
lines = [line.replace('np.complex,', 'complex,') for line in lines]

with open(file_path, 'w') as file:
    file.writelines(lines)

print(f"Modifications ready on path {file_path}")
