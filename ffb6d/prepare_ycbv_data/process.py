import os, sys
import numpy as np
from PIL import Image
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
import imageio
import argparse 
from tqdm import tqdm
import cv2

def render_mulimage_angles(obj_dir, data_id):

    anglesfileX = os.path.join(obj_dir, data_id+ "_anglesX.txt")
    anglesfileY = os.path.join(obj_dir, data_id+ "_anglesY.txt")
    anglesfileZ = os.path.join(obj_dir, data_id+ "_anglesZ.txt")

    new_img_file_angles = os.path.join(obj_dir, data_id+ "-pseudo_angles.png")
    

    anglesX = np.loadtxt(anglesfileX)
    anglesY = np.loadtxt(anglesfileY)
    anglesZ = np.loadtxt(anglesfileZ)
    anglesX[anglesX==360] = 255
    anglesX = (anglesX-anglesX[anglesX<255].min())*(254/(anglesX[anglesX<255].max()-anglesX[anglesX<255].min()))
    anglesY[anglesY==360] = 255
    anglesY = (anglesY-anglesY[anglesY<255].min())*(254/(anglesY[anglesY<255].max()-anglesY[anglesY<255].min()))
    anglesZ[anglesZ==360] = 255
    anglesZ = (anglesZ-anglesZ[anglesZ<255].min())*(254/(anglesZ[anglesZ<255].max()-anglesZ[anglesZ<255].min()))
    
    
    # combine three channels and save to a png image
    new_img_angles = np.dstack((anglesX, anglesY))
    new_img_angles = np.dstack((new_img_angles, anglesZ))

    cv2.imwrite(new_img_file_angles, new_img_angles)

    # remove the txt file in the angle directory
    os.system("rm " + anglesfileX)
    os.system("rm " + anglesfileY)
    os.system("rm " + anglesfileZ)


def render_mulimage_signed(obj_dir, data_id):

    signfileX = os.path.join(obj_dir, data_id+ "_signed_anglesX.txt")
    signfileY = os.path.join(obj_dir, data_id+ "_signed_anglesY.txt")
    signfileZ = os.path.join(obj_dir, data_id+ "_signed_anglesZ.txt")

    new_img_file_signed = os.path.join(obj_dir, data_id+ "-pseudo_signed.png")

    
    signX = np.loadtxt(signfileX)
    signX[signX==360] = 255
    signX = (signX-signX[signX<255].min())*(254/(signX[signX<255].max()-signX[signX<255].min()))
    signY = np.loadtxt(signfileY)
    signY[signY==360] = 255
    signY = (signY-signY[signY<255].min())*(254/(signY[signY<255].max()-signY[signY<255].min()))    
    signZ = np.loadtxt(signfileZ)
    signZ[signZ==360] = 255
    signZ = (signZ-signZ[signZ<255].min())*(254/(signZ[signZ<255].max()-signZ[signZ<255].min()))    
    
    # combine three channels and save to a png image

    new_img_sign = np.dstack((signX, signY))
    new_img_sign = np.dstack((new_img_sign, signZ))

    cv2.imwrite(new_img_file_signed, new_img_sign)

    # remove the txt file in the angle directory

    os.system("rm " + signfileX)
    os.system("rm " + signfileY)
    os.system("rm " + signfileZ)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', type=str, default='training_annotation.txt', help='Data labels')
    
    opt = parser.parse_args()
    example_file = opt.label
    
    example = np.loadtxt(example_file,dtype='str')
    for i in tqdm(range(1,4981)):

        data_dir = example[i].split(',')[0].split('/')[0]
        cat_id = example[i].split(',')[0].split('/')[1]
        obj_id = example[i].split(',')[0].split('/')[2]
        subdir = example[i].split(',')[0].split('/')[3]
        data_id = example[i].split(',')[0].split('/')[4].split('.')[0]
        obj_dir = os.path.join(data_dir, cat_id, obj_id)
        
        anglepath = obj_dir + "/points/" + data_id
        depthpath = obj_dir + "/depth/" + data_id
        imagepath = obj_dir + "/anglesimage/" + data_id
        new_img_path = obj_dir + "/pseudo/" + data_id
        if not os.path.isdir(obj_dir + "/anglesimage"):os.makedirs(obj_dir + "/anglesimage")
        if not os.path.isdir(obj_dir + "/pseudo"):os.makedirs(obj_dir + "/pseudo")


        anglesfile = anglepath + "_angles.txt"
        signfile = anglepath + "_signed_angles.txt"
        angles = np.loadtxt(anglesfile)
        sign = np.loadtxt(signfile)

        angleimg = Image.fromarray(angles).convert("L")
        signimg = Image.fromarray(sign).convert("L")

        anglename = imagepath + "_angles.png"
        signname = imagepath + "_signed_angles.png"

        angleimg.save(anglename)
        signimg.save(signname)
        
        depthfile = depthpath + "_0001.png"
         
        depthimg = np.array(Image.open(depthfile))
        new_img = np.zeros(shape=(depthimg.shape[0],depthimg.shape[1],3))
        new_img = new_img.astype(np.uint8)
        new_img[:,:,0] = depthimg
        new_img[:,:,1] = signimg
        new_img[:,:,2] = angleimg
        
        new_img = Image.fromarray(new_img)
        new_img_file = new_img_path + ".png"
        new_img.save(new_img_file)





