import os
import numpy as np
file_train = 'no_pseudo_angles.txt'
list_t = np.loadtxt(file_train,dtype=str)
root = 'YCB_Video_Dataset'
scene = []
i=0
for d in list_t[:18000]:
    
    scene = d.split('/')[0]
    img_id = d.split('/')[1]
    data1 = os.path.join(root,'data',scene,img_id+'-pseudo_angles.png')
    data2 = os.path.join(root,'data',scene,img_id+'-pseudo_signed.png')
    
    if not os.path.exists('data_new'): os.mkdir('data_new')
    if not os.path.exists(os.path.join('data_new',scene)): os.mkdir(os.path.join('data_new',scene))
    os.system("cp "+data1+" "+os.path.join('data_new',scene))
    os.system("cp "+data2+" "+os.path.join('data_new',scene))
