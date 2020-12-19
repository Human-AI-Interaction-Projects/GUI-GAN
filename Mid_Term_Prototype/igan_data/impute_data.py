# What: Data generation with SenseGen, which takes in user corrections and imputes corrected synthetic data
# Where: The code has been heavily modified and parts borrowed from: https://github.com/jsyoon0823/GAIN
# Why: Modifying existing code saves time for developing the major components of the system


# Usage example:
#   import imp_data
#   k = imp_data.impute_data() #see the arguments below.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import igan_data.data_utils
from igan_data.gain import gain_t
from igan_data.utils import binary_sampler

# hyperparameters: 1. data file (numpy) 2. data type (.mat or .csv) 3. vector to be imputed
#4. missing data rate during training 5. batch size 6. hint rate 7. alpha 8. no. of epochs
# more hyperparameters will be added after mid term

def impute_data(orig_data,
                data_type = '.mat',
                inp_data = [],
                miss_rate=0.3,
                batch_size = 128,
                hint_rate= 0.9,
                alpha = 100,
                iterations = 10000):
    #use original data used to train SenseGen and emulate missing data in it for GAIN training
    orig_data = np.transpose(orig_data)
    data_m = binary_sampler(1-miss_rate, orig_data.shape[0], orig_data.shape[1])
    train_data = orig_data.copy()
    train_data[data_m == 0] = np.nan
    gain_parameters = {'batch_size': batch_size,'hint_rate': hint_rate,'alpha': alpha,'iterations': iterations}
    t = np.transpose(np.tile(inp_data,[orig_data.shape[1],1]))
    print('Training...')
    imputed_data = gain_t(train_data, gain_parameters,t) #trains within seconds and returns several versions of imputed vector
    print('Training and Imputation Complete.')
    return imputed_data