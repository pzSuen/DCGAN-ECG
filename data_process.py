#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-10 21:23:09
# @Author  : Paul (pz.suen@gmail.com)

import numpy as np
from scipy.io import loadmat, savemat
import h5py

# 加载数据
def load_mat():
    ecg = loadmat("ecg.mat")
    ecg_data = ecg["RSegment_n"]
    print("ecg['RSegment_n'].shape=", ecg["RSegment_n"].shape)
    # print(ecg.keys())
    return ecg_data


def add_zeros(ecg_data):
    six_zeros = np.zeros((ecg_data.shape[0],6))
    zeros_add_ecg = np.append(ecg_data,six_zeros,axis=1)
    return zeros_add_ecg

# 数据划分并存储为h5文件
'''
	ecg_train = (8000, 250)
	ecg_valid = (999, 250)
	ecg_test = (1497, 250)
'''
def save_h5(ecg_data,ecg_file_name):
    f = h5py.File(ecg_file_name, 'w')
    f.create_dataset("ecg_train", data=ecg_data[:10000])
    f.create_dataset("ecg_test", data=ecg_data[10000:])

def load_h5(ecg_file_name):
    f = h5py.File(ecg_file_name,"r")
    a=f['ecg_train']
    b=f['ecg_test']
    print("a.shape:",a.shape)
    print("b.shape:",b.shape)


if __name__ == "__main__":
    ecg_data = load_mat()
    # save original data and test
    save_h5(ecg_data,"ecg_data_orig.h5")
    load_h5("ecg_data_orig.h5")

    # add six zeros and save and test
    zeros_add_ecg = add_zeros(ecg_data)
    assert zeros_add_ecg.shape == (ecg_data.shape[0],ecg_data.shape[1]+6),"The dimenssion is wrong."
    # print("zeros_add_ecg.shape:",zeros_add_ecg.shape)
    save_h5(zeros_add_ecg,"ecg_data_256.h5")
    load_h5("ecg_data_256.h5")



# '__header__': b'MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Fri Nov 10 10:19:42 2017', '__version__': '1.0', '__globals__': [], 'RSegment_n'
