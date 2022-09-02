import os
import glob
import os.path
import numpy as np
import sys 
sys.path.append('../..')

root = 'YCB_Video_Dataset/data'
root_syn = 'YCB_Video_Dataset/data_syn'
scenes_data = os.listdir(root)
scenes_syn = os.listdir(root_syn)

file1 = open('all_2.txt', 'r')
Lines = file1.readlines()

lines = np.unique(Lines)
print(len(lines))


'''for scene in scenes_data:
    for img_id in os.listdir(os.path.join(root, scene)):
        if img_id.endswith('.txt'):
            img_id = img_id.split('.')[0].split('-')[0]
            if os.path.isfile(os.path.join(root, scene, img_id+'-pseudo.png')):
                print(scene)   
                continue'''
                     
