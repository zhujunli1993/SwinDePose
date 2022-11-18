import cv2
import numpy as np
from PIL import Image
import os
import yaml
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import depth_map_utils_ycb as depth_map_utils
def fill_missing(
            dpt, cam_scale, scale_2_80m, fill_type='multiscale',
            extrapolate=False, show_process=False, blur_type='bilateral'
    ):
    dpt = dpt / cam_scale * scale_2_80m
    projected_depth = dpt.copy()
    if fill_type == 'fast':
        final_dpt = depth_map_utils.fill_in_fast(
            projected_depth, extrapolate=extrapolate, blur_type=blur_type,
            # max_depth=2.0
        )
    elif fill_type == 'multiscale':
        final_dpt, process_dict = depth_map_utils.fill_in_multiscale(
            projected_depth, extrapolate=extrapolate, blur_type=blur_type,
            show_process=show_process,
            max_depth=3.0
        )
    else:
        raise ValueError('Invalid fill_type {}'.format(fill_type))
    dpt = final_dpt / scale_2_80m * cam_scale
    return dpt
def derivative(I):
    #-Derivative x
    #Kx = np.array([[1,0,-1]])
    Kx = np.array([[-1,0,1]])
    Fx = ndimage.convolve(I, Kx)

    #-Derivative y
    #Ky = np.array([[1],[0],[-1]])
    Ky = np.array([[-1],[0],[1]])
    Fy = ndimage.convolve(I, Ky)
    #cv2.imwrite('/workspace/DATA/Linemod_preprocessed/Fx_0006.png',Fx)
    #cv2.imwrite('/workspace/DATA/Linemod_preprocessed/Fy_0006.png',Fy)
    
    #-Second Derivative xx
    Kxx = np.array([[0,0,0],[1,-2,1],[0,0,0]])
    Fxx = ndimage.convolve(I, Kxx)
    
    #-Second Derivative yy
    Kyy = np.array([[0,1,0],[0,-2,0],[0,1,0]])
    Fyy = ndimage.convolve(I, Kyy)
    
    Kxy = np.array([[0,0,0],[0,1,-1],[0,-1,1]])
    Fxy = ndimage.convolve(I, Kxy)
    #cv2.imwrite('/workspace/DATA/Linemod_preprocessed/Fxx_0006.png',Fxx)
    #cv2.imwrite('/workspace/DATA/Linemod_preprocessed/Fyy_0006.png',Fyy)
    #cv2.imwrite('/workspace/DATA/Linemod_preprocessed/Fxy_0006.png',Fxy)
    return Fx, Fy, Fxx, Fyy, Fxy

def derivative_large(I):
    #-Derivative x
    Kx = np.array([[-1,0,0,0,0,0,0,0,1]])
    Fx = ndimage.convolve(I, Kx)

    #-Derivative y
    Ky = np.array([[-1],[0],[0],[0],[0],[0],[0],[0],[1]])
    Fy = ndimage.convolve(I, Ky)
    
    #-Second Derivative xx
    Kxx = np.array([[0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0],
                    [1,0,0,0,-2,0,0,0,1],
                    [0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0]])
    Fxx = ndimage.convolve(I, Kxx)
    
    #-Second Derivative yy
    Kyy = np.array([[0,0,0,0,1,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,-2,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,1,0,0,0,0]])
    Fyy = ndimage.convolve(I, Kyy)
    
    Kxy = np.array([[0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,1,0,0,0,-1],
                    [0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,-1,0,0,0,1]])
    Fxy = ndimage.convolve(I, Kxy)
    return Fx, Fy, Fxx, Fyy, Fxy

def assign_curve(K, H):
    
    
    epislon = 1e-3
    
    out = np.zeros((K.shape[0], K.shape[1], 3))
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            if np.isnan(K[i,j]) or np.isnan(H[i,j]):
                out[i,j] = [0, 0, 0] # black
            if (K[i,j] <= 0+epislon and K[i,j] >= 0-epislon) and (H[i,j] <= 0+epislon and H[i,j] >= 0-epislon):
                out[i,j] = [0, 255, 0] # green
            if (K[i,j] <= 0+epislon and K[i,j] >= 0-epislon)  and H[i,j] > 0+epislon:
                out[i,j] = [255, 0, 0] # red
            if (K[i,j] <= 0+epislon and K[i,j] >= 0-epislon) and H[i,j] < 0-epislon:
                out[i,j] = [0, 0, 255] # blue
            if K[i,j] > 0+epislon and H[i,j] > 0+epislon:
                out[i,j] = [255, 255, 255] # white
            if K[i,j] > 0+epislon and H[i,j] < 0-epislon:
                out[i,j] = [255, 255, 0] # yellow
            if K[i,j] < 0-epislon:
                out[i,j] = [255, 0, 255] # purple
    return out           

def scale_pseudo(pseudo):
    # Scale the pseudo angles and signed angles to image range (0 ~ 255) 

    pseudo[pseudo==5] = 255
    pseudo[pseudo<255] = (pseudo[pseudo<255]-pseudo[pseudo<255].min())*(254/(pseudo[pseudo<255].max()-pseudo[pseudo<255].min()))
    
    return pseudo   

def scale_mean_curvature(pseudo):
    print(pseudo.max(), pseudo.min())
    pseudo = (pseudo-pseudo.min())*(255.0/(pseudo.max() - pseudo.min()))
    return pseudo             
                
def dpt_2_pcld(dpt, cam_scale, K):
    xmap = np.array([[j for i in range(640)] for j in range(480)])
    ymap = np.array([[i for i in range(640)] for j in range(480)])
    if len(dpt.shape) > 2:
        dpt = dpt[:, :, 0]
    dpt = dpt.astype(np.float32) / cam_scale
    msk = (dpt > 1e-8).astype(np.float32)
    row = (ymap - K[0][2]) * dpt / K[0][0]
    col = (xmap - K[1][2]) * dpt / K[1][1]
    dpt_3d = np.concatenate(
        (row[..., None], col[..., None], dpt[..., None]), axis=2
    )
    dpt_3d = dpt_3d * msk[:, :, None]
    return dpt_3d

K=np.array([[572.4114, 0.,         325.2611],
            [0.,        573.57043,  242.04899],
            [0.,        0.,         1.]])
meta_file = open(os.path.join('/workspace/DATA/Linemod_preprocessed/data','15', 'gt.yml'), "r")
meta_lst = yaml.safe_load(meta_file)

# for item_name in tqdm.tqdm(trainlist):
item_name = '0006'
with Image.open(os.path.join('/workspace/DATA/Linemod_preprocessed/data','15', "depth/{}.png".format(item_name))) as di:
    dpt_mm = np.array(di)

meta = meta_lst[int(item_name)]
meta = meta[0]

cam_scale = 1000.0
dpt_mm = fill_missing(dpt_mm, cam_scale, 1)

#dpt_mm = dpt_mm.copy().astype(np.uint16)
dpt_m = dpt_mm.astype(np.float32) / cam_scale
#dpt_m = gaussian_filter(dpt_m, sigma=5)
# import pdb;pdb.set_trace()
# dpt_m_show = ((dpt_m + 1.0) * 127).astype(np.uint8)
# cv2.imwrite('/workspace/DATA/Linemod_preprocessed/after_Gau_0006.png',dpt_m_show)
# if convert dpt_mm to uint8
#dpt_mm = dpt_mm.copy().astype(np.uint16)
dpt_m_show = ((dpt_m + 1.0) * 127).astype(np.uint8)
cv2.imwrite('/workspace/DATA/Linemod_preprocessed/no_Gau_0006.png',dpt_m_show)

Fx, Fy, Fxx, Fyy, Fxy = derivative_large(dpt_m)
Fx = np.array(Fx)
Fy = np.array(Fy)
Fxx = np.array(Fxx)
Fyy = np.array(Fyy)
Fxy = np.array(Fxy)
K = (Fxx * Fyy - np.power(Fxy, 2)) / np.power((1 + np.power(Fx, 2) + np.power(Fy, 2)), 2)
H_2 = ((1 + np.power(Fx, 2)) * Fyy - 2 * Fx * Fy * Fxy + (1 + np.power(Fy, 2)) * Fxx) / np.power(1 + np.power(Fx, 2) + np.power(Fy, 2), 3/2)
H = H_2 / 2
# K[np.isnan(K)] = 0.0
# H[np.isinf(H)] = 0.0
out = assign_curve(K, H)
#print(K.max(), K.min())
#print(H.max(), H.min())
H = scale_mean_curvature(H)

cv2.imwrite('/workspace/DATA/Linemod_preprocessed/curClass_0006_mean_curve.png',H)
cv2.imwrite('/workspace/DATA/Linemod_preprocessed/curClass_0006.png',out)
#nrm_map = normalSpeed.depth_normal(dpt_mm, K[0][0], K[1][1], 5, 2000, 20, False)

# if True:
#     show_nrm_map = ((nrm_map + 1.0) * 127).astype(np.uint8)
#     img_file = os.path.join('/workspace/DATA/Linemod_preprocessed/data',args.cls_num,'vis_nrm_{}.png'.format(item_name))
#     cv2.imwrite(img_file, show_nrm_map)
# dpt_m = dpt_mm.astype(np.float32) / cam_scale
# dpt_xyz = dpt_2_pcld(dpt_m, 1.0, K)
# dpt_xyz[np.isnan(dpt_xyz)] = 0.0
# dpt_xyz[np.isinf(dpt_xyz)] = 0.0
# pcd = o3d.io.read_point_cloud("data/1.pcd")
# surface_curvature = caculate_surface_curvature(pcd, radius=0.003)
# print(surface_curvature[:10]) 
