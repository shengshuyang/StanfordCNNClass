# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 16:16:46 2016

@author: shuyang

This is a short snippet that shows how to write images to lmdb
""" 

import caffe
import lmdb
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mp

import caffe.proto.caffe_pb2
from caffe.io import array_to_datum


def write_images_to_lmdb(img_dir, db_name):

    for root, dirs, files in os.walk(img_dir, topdown = False):

        if root != img_dir:
            continue

        map_size = 64*64*3*2*len(files)
        env = lmdb.Environment(db_name, map_size=map_size)
        txn = env.begin(write=True,buffers=True)
        
        for idx, name in enumerate(files):
            X = mp.imread(os.path.join(root, name))
            y = 1
            datum = array_to_datum(X,y)
            str_id = '{:08}'.format(idx)
            # The encode is only essential in Python 3
            txn.put(str_id.encode('ascii'), datum.SerializeToString())
        
    txn.commit()
    env.close()
    print " ".join(["Writing to", db_name, "done!"])


if __name__ == '__main__':
    write_images_to_lmdb(img_dir = 'patches', db_name = 'shadow_patches')




