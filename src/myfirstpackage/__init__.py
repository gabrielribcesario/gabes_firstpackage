import numpy as np
from copy import deepcopy

#https://stackoverflow.com/a/23093013
py_files = ['np', 'deepcopy']
__all__ = py_files
"""
from os import listdir, abspath
from os.path import abspath, join, isfile, dirname

# get folder name of __init__.py
init_dir = dirname(abspath(__file__))
# get all python files
for file_name in listdir(init_dir):
  if isfile(join(init_dir, file_name)) and ".py" in file_name and not ".pyc" in file_name:
    py_files.append(file_name.replace(".py", ""))
# remove this __init__ file from the list
py_files.remove("__init__")
"""
