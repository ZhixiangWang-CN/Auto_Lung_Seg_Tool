import numpy as np
import pandas as pd
import pydicom as dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt
import nrrd
from skimage import measure, morphology
from scipy import ndimage as ndi
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import skimage, os
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border, mark_boundaries
from skimage import data
from scipy import ndimage as ndi
import matplotlib.pyplot as plt


def get_segmented_lungs(raw_im,shape_n, plot=False):
    '''
    Original function changes input image (ick!)
    '''
    im = raw_im.copy()
    rate = shape_n/512
    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    if plot == True:
        f, plots = plt.subplots(8, 1, figsize=(5, 40))
    '''
    Step 1: Convert into a binary image. 
    '''
    binary = im < -400
    if plot == True:
        plots[0].axis('off')
        plots[0].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)
    if plot == True:
        plots[1].axis('off')
        plots[1].imshow(cleared, cmap=plt.cm.bone)
    '''
    Step 3: Label the image.
    '''
    label_image = label(cleared)
    if plot == True:
        plots[2].axis('off')
        plots[2].imshow(label_image, cmap=plt.cm.bone)
    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    if plot == True:
        plots[3].axis('off')
        plots[3].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is 
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    if plot == True:
        plots[4].axis('off')
        plots[4].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is 
    to keep nodules attached to the lung wall.
    '''
    selem = disk(10)
    binary = binary_closing(binary, selem)
    if plot == True:
        plots[5].axis('off')
        plots[5].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    if plot == True:
        plots[6].axis('off')
        plots[6].imshow(binary, cmap=plt.cm.bone)
    num_one = np.sum(binary)
    if num_one < int(2000*rate):
        center_mask = np.ones(binary.shape)
        w_c = center_mask.shape[0] // 2
        h_c = center_mask.shape[1] // 2
        center_mask[w_c - int(50*rate):w_c + int(50*rate), h_c - int(100*rate):h_c -int(25*rate)] = 0
        center_mask[w_c - int(50*rate):w_c + int(50*rate), h_c +int(25*rate) :h_c + int(100*rate)] = 0
        masked_surrounding = center_mask*binary
        n_surrounding = np.sum(masked_surrounding)
        if n_surrounding>int(100*rate):
            binary = morphology.remove_small_objects(binary, int(600*rate), connectivity=2, in_place=True)
        else:
            binary = morphology.remove_small_objects(binary, int(200*rate), connectivity=2, in_place=True)
    if plot == True:
        plots[5].axis('off')
        plots[5].imshow(center_mask, cmap=plt.cm.bone)
    # # num_one = np.sum(binary)


    '''
    Step 8: Superimpose the binary mask on the input image.
    '''
    get_high_vals = binary == 0
    im[get_high_vals] = 0
    if plot == True:
        plots[7].axis('off')
        plots[7].imshow(im, cmap=plt.cm.bone)
        plt.show()
    return binary


def cut_lung_from_background(data,shape_n=512):
    mask = np.zeros(data.shape,dtype=np.uint8)
    # masked_img = np.zeros(data.shape)

    shape = data.shape
    data[int(shape_n*(360/512)):, :,:] = -1000
    for i in range(shape[-1]):
        # print(i)
        # i=90
        img = data[:,:,i]
        mk=get_segmented_lungs(img,shape_n,False)
        mask[:,:,i]=mk
        # masked_img[:,:,i]=mk_img
    return mask

if __name__ == "__main__":
    data,head = nrrd.read('')
    mask = cut_lung_from_background(data)
    nrrd.write('seg_lung.nrrd',mask)
    print("finish")
