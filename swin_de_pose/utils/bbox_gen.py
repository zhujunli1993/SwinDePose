import numpy as np
import os
import glob
import cv2
cls = '15'
folder = os.path.join('/workspace/DATA/Linemod_preprocessed/data', cls, 'mask')
max_h=240
max_w=320
# for file in glob.glob(folder+'/*.png'):
#     img = cv2.imread(file)
#     gray =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     ret,binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
#     _, contours, _ = cv2.findContours(binary,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     x,y,w,h = cv2.boundingRect(contours[0])
#     if w > max_w:
#         max_w = w
#     if h > max_h:
#         max_h = h
# bbox = []
# for file in glob.glob(folder+'/*.png'):

#     num = file.split('/')[7].split('.')[0]
#     img = cv2.imread(file)
#     gray =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     ret,binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
#     _, contours, _ = cv2.findContours(binary,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     x,y,w,h = cv2.boundingRect(contours[0])
#     cen_w, cen_h = int(x+w/2), int(y+h/2)
#     l_w, l_h = int(cen_w-max_w/2), int(cen_h-max_h/2)
#     h_w, h_h = l_w + max_w, l_h + max_h
#     if l_w < 0:
#         l_w = 0
#         h_w = max_w
#     if l_h < 0:
#         l_h = 0
#         h_h = max_h
#     if h_w > 640:
#         h_w = 640
#         l_w = max_w
#     if h_h > 480:
#         h_h = 480
#         l_h = max_h
        
#     box = [num, l_w, l_h, h_w, h_h]
#     bbox.append(box)
# np.save(os.path.join(folder, 'bbox.npy'), bbox)
bbox = np.load(os.path.join(folder, 'bbox.npy'))
for v in bbox:
    
    num = v[0]
    l_w = int(v[1])
    l_h = int(v[2])
    h_w = int(v[3])
    h_h = int(v[4])
    
    # rgb = cv2.imread(os.path.join(os.path.join('/workspace/DATA/Linemod_preprocessed/data', cls, 'rgb'), num+'.png'))
    # im_write = rgb[l_h:h_h,l_w:h_w, :]
    # cv2.imwrite(os.path.join('/workspace/DATA/Linemod_preprocessed/data/15/','crop',num+'.png'),im_write)
    
    dep_mm = cv2.imread(os.path.join(os.path.join('/workspace/DATA/Linemod_preprocessed/data', cls, 'depth'), num+'.png'))
    dpt_mm_rgb = dep_mm.copy()
    second_min = np.unique(dpt_mm_rgb)[1]
    index = np.where(dpt_mm_rgb==0)
    factor = 254. / (np.max(dpt_mm_rgb)-second_min)
    dpt_mm_rgb = factor * (dpt_mm_rgb - second_min)
    dpt_mm_rgb[index] = 255
    dpt_mm_crop = dpt_mm_rgb[l_h:h_h,l_w:h_w]
    cv2.imwrite(os.path.join('/workspace/DATA/Linemod_preprocessed/data/15/','crop_depth',num+'.png'),dpt_mm_crop)
    cv2.imwrite(os.path.join('/workspace/DATA/Linemod_preprocessed/data/15/','uncrop_depth',num+'.png'),dpt_mm_rgb)



# def get_crop(bbox_folder, item_name, max_w=240, max_h=320):
#     bbox = np.load(os.path.join(bbox_folder, 'bbox.npy'))
#     boolArr = (bbox[:,0] == item_name)
#     result = np.where(boolArr)
#     index = int(result[0])
#     loc = bbox[index]
#     l_w, l_h = int(loc[1]), int(loc[2])
#     cen_w, cen_h = int(loc[5]), int(loc[6])
#     h_w, h_h = l_x + max_w, l_y + max_h
#     return l_w, l_h, h_w, h_h, cen_w, cen_h

