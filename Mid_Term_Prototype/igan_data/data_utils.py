# What: Data loading utils for SenseGen
# Where: https://github.com/nesl/sensegen
# Why: Modifying existing code saves time for developing the major components of the system


"""
Author: Moustafa Alzantot (malzantot@ucla.edu)
All rights reserved Networked and Embedded Systems Lab (NESL), UCLA.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
import os
import sys
from scipy.io import loadmat
from zipfile import ZipFile


#dataset is a zip file
def load_training_data(data_file, data_type='.mat'):
    """ Returns a matrix of training data.
    shape of result = (n_exp, len)
    """
    print('Loading dataset...')
    if(data_type=='.csv'):
        t = [i for i, letter in enumerate(data_file) if letter == '/']
        with ZipFile(data_file, 'r') as zipObj:
            zipObj.extractall(data_file[:t[-1]+1])
        data_dir = data_file[:-4]
        data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        data_list = []
        for fname in data_files:
            exp_data =  np.genfromtxt(os.path.join(data_dir, fname), delimiter=",")
            exp_data = (exp_data - np.mean(exp_data)) / np.std(exp_data) # normalize for numerical stability
            data_list.append(exp_data)
        data = np.stack(data_list)
        print('Loaded dataset.')
        return data
    else:
        with ZipFile(data_file, 'r') as zipObj:
            zipObj.extractall(os.path.dirname(data_file))
        data_dir = os.path.join(os.path.dirname(data_file), zipObj.namelist()[0][:-1])
        data_files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
        data_list = []
        for fname in data_files:
            exp_data = loadmat(os.path.join(data_dir, fname))['val'].astype(np.float32).ravel()
            exp_data = (exp_data - np.mean(exp_data)) / np.std(exp_data) # normalize for numerical stability
            data_list.append(exp_data)
            os.remove(os.path.join(data_dir, fname))
        os.rmdir(data_dir)
        data = np.stack(data_list)
        print('Loaded dataset.')
        return data
    
          

class DataLoader(object):
    def __init__(self, data, batch_size=128, num_steps=1):
        self.batch_size = batch_size
        self.n_data, self.seq_len = data.shape
        #num_batches = (self.n_data // self.batch_size) * self.batch_size
        self._data = data[:self.batch_size , :]
        
        self.num_steps = num_steps
        self._data = self._data.reshape((self.batch_size, self.seq_len, 1))
        self._reset_pointer()

    def _reset_pointer(self):
        self.pointer = 0

    def reset(self):
        self._reset_pointer()

    def has_next(self):
        return self.pointer + self.num_steps < self.seq_len - 1

    def next_batch(self):
        batch_xs = self._data[:, self.pointer:self.pointer+self.num_steps, :]
        batch_ys = self._data[:, self.pointer+1:self.pointer+self.num_steps+1, :]
        self.pointer = self.pointer + self.num_steps
        return batch_xs, batch_ys
        

