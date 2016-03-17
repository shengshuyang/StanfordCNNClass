import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from math import sqrt
from os.path import expanduser


def extract_patches(path, filename, out_path, patch_size,  stride,  visualize):
    img = mpimg.imread(path+filename)
    nRows, nCols, nColor = img.shape
    psx, psy = patch_size
    
    patches = []
    for r in xrange(psy/2+1, nRows - psy/2 - 1, stride):
        for c in xrange(psx/2+1, nCols - psx/2 - 1, stride):
           patches.append(img[r-psy/2 : r + psy/2, c-psx/2 : c+psx/2, :]) 
    grid_size = int(sqrt(len(patches)))
    
    name, ext = os.path.splitext(filename)
    for pos in xrange(len(patches)):
        plt.imsave(out_path + name + "_" + str(pos) + ext, patches[pos])
    
    if not visualize:
        return
        
    for pos in xrange(len(patches)):
        if pos + 1 < grid_size ** 2:
            plt.subplot(grid_size, grid_size, pos+1)
            plt.imshow(patches[pos])
            plt.axis('off')

if __name__ == "__main__":
    home = expanduser("~")
    nyu_path = home+'/IGNORE_NYU/jpgs/'
    #extract_patches(, [16,16], 100, True)
    for root, dirs, files in os.walk(nyu_path, topdown=False):
        for filename in files:
            extract_patches(nyu_path, filename, nyu_path+"/patches/", [64,64], 100, False)

    