import numpy as np
import tqdm
def scale_pseudo(pseudo, idx):
        
    # Scale the pseudo angles and signed angles to image range (0 ~ 255) 
    pseudo[:,:,0][pseudo[:,:,0]==360] = 255
    if (pseudo[:,:,0][pseudo[:,:,0]<255].max()-pseudo[:,:,0][pseudo[:,:,0]<255].min())==0:
        print(idx)
    if (pseudo[:,:,1][pseudo[:,:,1]<255].max()-pseudo[:,:,1][pseudo[:,:,1]<255].min())==0:
        print(idx)
    if (pseudo[:,:,2][pseudo[:,:,2]<255].max()-pseudo[:,:,2][pseudo[:,:,2]<255].min())==0:
        print(idx)
    pseudo[:,:,0][pseudo[:,:,0]<255] = (pseudo[:,:,0][pseudo[:,:,0]<255]-pseudo[:,:,0][pseudo[:,:,0]<255].min())*(254/(pseudo[:,:,0][pseudo[:,:,0]<255].max()-pseudo[:,:,0][pseudo[:,:,0]<255].min()))
    pseudo[:,:,1][pseudo[:,:,1]==360] = 255
    pseudo[:,:,1][pseudo[:,:,1]<255] = (pseudo[:,:,1][pseudo[:,:,1]<255]-pseudo[:,:,1][pseudo[:,:,1]<255].min())*(254/(pseudo[:,:,1][pseudo[:,:,1]<255].max()-pseudo[:,:,1][pseudo[:,:,1]<255].min()))
    pseudo[:,:,2][pseudo[:,:,2]==360] = 255
    pseudo[:,:,2][pseudo[:,:,2]<255] = (pseudo[:,:,2][pseudo[:,:,2]<255]-pseudo[:,:,2][pseudo[:,:,2]<255].min())*(254/(pseudo[:,:,2][pseudo[:,:,2]<255].max()-pseudo[:,:,2][pseudo[:,:,2]<255].min()))
    
    # pseudo[:,:,0][pseudo[:,:,0]==360] = 255
    # pseudo[:,:,0][pseudo[:,:,0]<255] = pseudo[:,:,0][pseudo[:,:,0]<255]*254.0/180.0
    # pseudo[:,:,1][pseudo[:,:,1]==360] = 255
    # pseudo[:,:,1][pseudo[:,:,1]<255] = pseudo[:,:,1][pseudo[:,:,1]<255]*254.0/180.0
    # pseudo[:,:,2][pseudo[:,:,2]==360] = 255
    # pseudo[:,:,2][pseudo[:,:,2]<255] = pseudo[:,:,2][pseudo[:,:,2]<255]*254.0/180.0
    
    return pseudo
    
ren_path = '/workspace/DATA/Linemod_preprocessed/fuse_nrm/driller/file_list.txt'
contents = np.loadtxt(ren_path, dtype='str')
for file in tqdm.tqdm(contents):
    values = np.load(file)
    angle = values['angles']
    sed_angles = scale_pseudo(angle,file)
    if len(np.unique(sed_angles)) <= 1:
        print(file)
        
ren_path = '/workspace/DATA/Linemod_preprocessed/renders_nrm/driller/file_list.txt'
contents = np.loadtxt(ren_path, dtype='str')
for file in tqdm.tqdm(contents):
    values = np.load(file)
    angle = values['angles']
    sed_angles = scale_pseudo(angle,file)
    if len(np.unique(sed_angles)) <= 1:
        print(file)