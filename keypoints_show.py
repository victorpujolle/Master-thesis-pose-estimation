import numpy as np
import cv2
from utils import *
import os
import argparse
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection

def main():
    nb_kp = 8
    nb_img = 10
    path = '..\\..\\Dataset_Generation\\Blender\\result_virtual_dataset_256\\'
    test_file_name = '..\\..\\Dataset_Generation\\Blender\\txt_files\\valid_virtual_256.txt'
    #path = '..\\..\\Dataset_Generation\\Blender\\Rendus\\'
    #test_file_name = '..\\..\\Dataset_Generation\\Blender\\txt_files\\test1.txt'

    HM = np.zeros([64,64,3])
    for i in range(nb_img):
        name_img, joints, bb = v2_read_text_file(test_file_name, i + 1)
        print(os.path.join(path, name_img))
        img = cv2.imread(os.path.join(path, name_img))

        heatmap = readHM(filepath=path, name=name_img)
        [W,score] = findWMax(HM)
        response = np.sum(heatmap, 2)
        max_value = np.amax(response)
        min_value = np.amin(response)
        response = (response - min_value) / (max_value - min_value)
        cmap = plt.get_cmap('viridis')
        mapIm = np.delete(cv2.resize(cmap(response), (256, 256)), 3, 2)
        imgtoshow = 0.5*(img + mapIm*255) /255


        joints[:, 1] = 256 - joints[:, 1]
        bb[:, 1] = 256 - bb[:, 1]

        for k in range(8):
            #cv2.circle(img, (joints[k,0],joints[k,1]), 4,(255,0,0),-1)
            cv2.circle(imgtoshow, (bb[k, 0], bb[k, 1]), 4, (0, 0, 255), -1)

        cv2.imshow('image', imgtoshow)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
