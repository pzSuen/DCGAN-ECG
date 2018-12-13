#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-10 21:52:54
# @Author  : Paul (pz.suen@gmail.com)
# @Link    : ${link}
# @Version : $Id$

import numpy as np
import h5py

f = h5py.File("ecg_data.h5", 'r')
ecg_train = f['ecg_train']
ecg_valid = f['ecg_valid']
ecg_test = f['ecg_test']

print("ecg_train =", ecg_train.shape)
print("ecg_valid =", ecg_valid.shape)
print("ecg_test =", ecg_test.shape)
