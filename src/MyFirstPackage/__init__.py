from copy import deepcopy
from os import listdir
from os.path import abspath, dirname, isfile, join
import numpy as np

# get location of __init__.py and get folder name of __init__.py
init_dir = listdir(dirname(abspath(__file__)))
init_dir.remove("__init__.py")

modulename = ['deepcopy', 'listdir', 'abspath', 'dirname', 'isfile', 'join', 'np']
for file_name in init_dir:
  if isfile(join(init_dir, file_name)) and ".py" in file_name and not ".pyc" in file_name:
    modulename.append(filename.replace('.py', ''))
    
__all__ = modulename
