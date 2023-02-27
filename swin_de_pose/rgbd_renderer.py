#!/usr/bin/env python3
import os
import pickle as pkl
import numpy as np
from plyfile import PlyData
import ctypes as ct
import cv2
import random
from random import randint
from random import shuffle
from tqdm import tqdm
from scipy import stats
from glob import glob
from argparse import ArgumentParser
import concurrent.futures
import time
import math
import depth_map_utils_ycb as depth_map_utils
import normalSpeed
parser = ArgumentParser()
parser.add_argument(
    "--cls", type=str, default="ape",
    help="Target object from {ape, benchvise, cam, can, cat, driller, duck, \
    eggbox, glue, holepuncher, iron, lamp, phone} (default ape)"
)
parser.add_argument(
    '--render_num', type=int, default=70000,
    help="Number of images you want to generate."
)
parser.add_argument(
    '--DEBUG', action="store_true",
    help="To show the generated images or not."
)
parser.add_argument(
    '--vis', action="store_true",
    help="visulaize generated images."
)
args = parser.parse_args()
DEBUG = args.DEBUG


def check_dir(pth):
    if not os.path.exists(pth):
        os.system("mkdir -p {}".format(pth))


OBJ_ID_DICT = {
    'ape': 1,
    'benchvise': 2,
    'cam': 4,
    'can': 5,
    'cat': 6,
    'driller': 8,
    'duck': 9,
    'eggbox': 10,
    'glue': 11,
    'holepuncher': 12,
    'iron': 13,
    'lamp': 14,
    'phone': 15,
}
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


def pseudo_nrm_angle(nrm_map):
    height=480
    width=640
    # Set up-axis 
    x_up = np.array([1.0, 0.0, 0.0])
    y_up = np.array([0.0, 1.0, 0.0])
    z_up = np.array([0.0, 0.0, 1.0])

    angle_x = []
    # signed_x = []
    angle_y = []
    # signed_y = []
    angle_z = []
    # signed_z = []
    
    for i in range(0, height):
        for j in range(0, width):
            if sum(nrm_map[i, j])==0.0:
                angle_x.append(360.0)
                angle_y.append(360.0)
                angle_z.append(360.0)
                continue
            else:
                nrm_c = nrm_map[i, j]
                epsilon = 1E-6
                value_x = nrm_c[0]*x_up[0] + nrm_c[1]*x_up[1] + nrm_c[2]*x_up[2]
                value_y = nrm_c[0]*y_up[0] + nrm_c[1]*y_up[1] + nrm_c[2]*y_up[2]
                value_z = nrm_c[0]*z_up[0] + nrm_c[1]*z_up[1] + nrm_c[2]*z_up[2]
                if value_x > (1.0 - epsilon):
                    value_x = 1.0
                elif value_x < (epsilon - 1.0):
                    value_x = -1.0 
                angle_x.append(np.arccos(value_x)*180.0/math.pi)
                
                if value_y > (1.0 - epsilon):
                    value_y = 1.0
                elif value_y < (epsilon - 1.0):
                    value_y = -1.0 
                angle_y.append(np.arccos(value_y)*180.0/math.pi)
                
                if value_z > (1.0 - epsilon):
                    value_z = 1.0
                elif value_z < (epsilon - 1.0):
                    value_z = -1.0 
                angle_z.append(np.arccos(value_z)*180.0/math.pi)
    angle_x = np.reshape(angle_x, [height, width])
    # signed_x = np.reshape(signed_x, [height, width])
    angle_y = np.reshape(angle_y, [height, width])
    # signed_y = np.reshape(signed_y, [height, width])
    angle_z = np.reshape(angle_z, [height, width])            
    
    new_img_angles = np.dstack((angle_x, angle_y))
    new_img_angles = np.dstack((new_img_angles, angle_z))
    return new_img_angles            
def pseudo_gen(dpt_xyz):
    height=480
    width=640
    # Set up-axis 
    x_up = np.array([1.0, 0.0, 0.0])
    y_up = np.array([0.0, 1.0, 0.0])
    z_up = np.array([0.0, 0.0, 1.0])

    angle_x = []
    signed_x = []
    angle_y = []
    signed_y = []
    angle_z = []
    signed_z = []
    dpt_xyz = np.reshape(dpt_xyz,(height, width, 3))
    for i in range(0, height):
        for j in range(1, width):
            p_2 = dpt_xyz[i, j]
            p_1 = dpt_xyz[i, j-1]
            if p_2[0]+p_2[1]+p_2[2]==0.0 or p_1[0]+p_1[1]+p_1[2]==0.0:
                angle_x.append(360.0)
                angle_y.append(360.0)
                angle_z.append(360.0)
                signed_x.append(360.0)
                signed_y.append(360.0)
                signed_z.append(360.0)
                continue
            else:
                difference = p_2 - p_1
                difference_lengh = np.sqrt(math.pow(difference[0],2)+math.pow(difference[1],2)+math.pow(difference[2],2))
                epsilon = 1E-6
                if difference_lengh < epsilon:
                    angle_x.append(360.0)
                    angle_y.append(360.0)
                    angle_z.append(360.0)
                    signed_x.append(360.0)
                    signed_y.append(360.0)
                    signed_z.append(360.0)
                    continue
                else:
                    value_x = (difference[0]*x_up[0] + difference[1]*x_up[1] + difference[2]*x_up[2]) / difference_lengh
                    value_y = (difference[0]*y_up[0] + difference[1]*y_up[1] + difference[2]*y_up[2]) / difference_lengh
                    value_z = (difference[0]*z_up[0] + difference[1]*z_up[1] + difference[2]*z_up[2]) / difference_lengh
                    if value_x > (1.0 - epsilon):
                        value_x = 1.0
                    elif value_x < (epsilon - 1.0):
                        value_x = -1.0 
                    angle_x.append(np.arccos(value_x)*180.0/math.pi)
                    
                    if value_y > (1.0 - epsilon):
                        value_y = 1.0
                    elif value_y < (epsilon - 1.0):
                        value_y = -1.0 
                    angle_y.append(np.arccos(value_y)*180.0/math.pi)
                    
                    if value_z > (1.0 - epsilon):
                        value_z = 1.0
                    elif value_z < (epsilon - 1.0):
                        value_z = -1.0 
                    angle_z.append(np.arccos(value_z)*180.0/math.pi)
                    
                    if j == 1:
                        signed_x.append(360.0)
                        signed_y.append(360.0)
                        signed_z.append(360.0)
                        continue
                    else:
                        
                        p_0 = dpt_xyz[i, j-2]
                    
                        if p_0[0]+p_0[1]+p_0[2]==0.0:
                            signed_x.append(360.0)
                            signed_y.append(360.0)
                            signed_z.append(360.0)
                            continue
                        else:
                            dot_prod = difference[0]*(p_1[0]-p_0[0]) + difference[1]*(p_1[1]-p_0[1]) + difference[2]*(p_1[2]-p_0[2])
                            if dot_prod >= 0.0:
                                signed_x.append(math.acos(value_x)*180.0/math.pi)
                            else:
                                signed_x.append(-1*math.acos(value_x)*180.0/math.pi)
                            if dot_prod >= 0.0:
                                signed_y.append(math.acos(value_y)*180.0/math.pi)
                            else:
                                signed_y.append(-1*math.acos(value_y)*180.0/math.pi)
                            if dot_prod >= 0.0:
                                signed_z.append(math.acos(value_z)*180.0/math.pi)
                            else:
                                signed_z.append(-1*math.acos(value_z)*180.0/math.pi)
        angle_x.append(360.0)
        angle_y.append(360.0)
        angle_z.append(360.0)
        signed_x.append(360.0)
        signed_y.append(360.0)
        signed_z.append(360.0)
        
    angle_x = np.reshape(angle_x, [height, width])
    signed_x = np.reshape(signed_x, [height, width])
    angle_y = np.reshape(angle_y, [height, width])
    signed_y = np.reshape(signed_y, [height, width])
    angle_z = np.reshape(angle_z, [height, width])
    signed_z = np.reshape(signed_z, [height, width])
    new_img_angles = np.dstack((angle_x, angle_y))
    new_img_angles = np.dstack((new_img_angles, angle_z))
    new_img_signed = np.dstack((signed_x, signed_y))
    new_img_signed = np.dstack((new_img_signed, signed_z))
    return new_img_angles, new_img_signed

class LineModRenderDB():
    def __init__(self, cls_type, render_num=10, rewrite=False):
        self.h, self.w = 480, 640
        self.K = np.array([[700., 0., 320.],
                           [0., 700., 240.],
                           [0., 0., 1.]])

        self.cls_type = cls_type
        self.cls_id = OBJ_ID_DICT[cls_type]

        self.linemod_dir = './Linemod_preprocessed'
        self.render_dir = os.path.join(self.linemod_dir, 'renders_nrm', cls_type)
        check_dir(self.render_dir)
        self.render_num = render_num
        RT_pth = os.path.join('sampled_poses', '{}_sampled_RTs.pkl'.format(cls_type))
        self.RT_lst = pkl.load(open(RT_pth, 'rb'))

        so_p = './rastertriangle_so.so'
        self.dll = np.ctypeslib.load_library(so_p, '.')

        self.bg_img_pth_lst = glob("SUN2012pascalformat/JPEGImages/*.jpg")

        random.seed(19763)
        if render_num < len(self.RT_lst):
            random.shuffle(self.RT_lst)
            self.RT_lst = self.RT_lst[:render_num]

        print("begin loading {} render set:".format(cls_type))

        b_mdl_p = os.path.join(
            self.linemod_dir, 'models', 'obj_%02d.ply' % self.cls_id
        )
        self.npts, self.xyz, self.r, self.g, self.b, self.n_face, self.face = self.load_ply_model(b_mdl_p)
        self.face = np.require(self.face, 'int32', 'C')
        self.r = np.require(np.array(self.r), 'float32', 'C')
        self.g = np.require(np.array(self.g), 'float32', 'C')
        self.b = np.require(np.array(self.b), 'float32', 'C')

    def load_ply_model(self, model_path):
        ply = PlyData.read(model_path)
        data = ply.elements[0].data
        x = data['x']
        y = data['y']
        z = data['z']
        r = data['red']
        g = data['green']
        b = data['blue']
        face_raw = ply.elements[1].data
        face = []
        for item in face_raw:
            face.append(item[0])

        n_face = len(face)
        face = np.array(face).flatten()

        n_pts = len(x)
        xyz = np.stack([x, y, z], axis=-1) / 1000.0

        return n_pts, xyz, r, g, b, n_face, face

    def gen_pack_zbuf_render(self):
        pth_lst = []

        for idx, RT in tqdm(enumerate(self.RT_lst)):
            h, w = self.h, self.w
            R, T = RT[:, :3], RT[:, 3]
            K = self.K

            new_xyz = self.xyz.copy()
            new_xyz = np.dot(new_xyz, R.T) + T
            p2ds = np.dot(new_xyz.copy(), K.T)
            p2ds = p2ds[:, :2] / p2ds[:, 2:]
            p2ds = np.require(p2ds.flatten(), 'float32', 'C')

            zs = np.require(new_xyz[:, 2].copy(), 'float32', 'C')
            zbuf = np.require(np.zeros(h*w), 'float32', 'C')
            rbuf = np.require(np.zeros(h*w), 'int32', 'C')
            gbuf = np.require(np.zeros(h*w), 'int32', 'C')
            bbuf = np.require(np.zeros(h*w), 'int32', 'C')

            self.dll.rgbzbuffer(
                ct.c_int(h),
                ct.c_int(w),
                p2ds.ctypes.data_as(ct.c_void_p),
                new_xyz.ctypes.data_as(ct.c_void_p),
                zs.ctypes.data_as(ct.c_void_p),
                self.r.ctypes.data_as(ct.c_void_p),
                self.g.ctypes.data_as(ct.c_void_p),
                self.b.ctypes.data_as(ct.c_void_p),
                ct.c_int(self.n_face),
                self.face.ctypes.data_as(ct.c_void_p),
                zbuf.ctypes.data_as(ct.c_void_p),
                rbuf.ctypes.data_as(ct.c_void_p),
                gbuf.ctypes.data_as(ct.c_void_p),
                bbuf.ctypes.data_as(ct.c_void_p),
            )

            zbuf.resize((h, w))
            msk = (zbuf > 1e-8).astype('uint8')
            if len(np.where(msk.flatten() > 0)[0]) < 500:
                continue
            zbuf *= msk.astype(zbuf.dtype)  # * 1000.0

            bbuf.resize((h, w)), rbuf.resize((h, w)), gbuf.resize((h, w))
            bgr = np.concatenate((bbuf[:, :, None], gbuf[:, :, None], rbuf[:, :, None]), axis=2)
            bgr = bgr.astype('uint8')

            bg = None
            len_bg_lst = len(self.bg_img_pth_lst)
            while bg is None or len(bg.shape) < 3:
                bg_pth = self.bg_img_pth_lst[randint(0, len_bg_lst-1)]
                bg = cv2.imread(bg_pth)
                if len(bg.shape) < 3:
                    bg = None
                    continue
                bg_h, bg_w, _ = bg.shape
                if bg_h < h:
                    new_w = int(float(h) / bg_h * bg_w)
                    bg = cv2.resize(bg, (new_w, h))
                bg_h, bg_w, _ = bg.shape
                if bg_w < w:
                    new_h = int(float(w) / bg_w * bg_h)
                    bg = cv2.resize(bg, (w, new_h))
                bg_h, bg_w, _ = bg.shape
                if bg_h > h:
                    sh = randint(0, bg_h-h)
                    bg = bg[sh:sh+h, :, :]
                bg_h, bg_w, _ = bg.shape
                if bg_w > w:
                    sw = randint(0, bg_w-w)
                    bg = bg[:, sw:sw+w, :]

            msk_3c = np.repeat(msk[:, :, None], 3, axis=2)
            bgr = bg * (msk_3c <= 0).astype(bg.dtype) + bgr * (msk_3c).astype(bg.dtype)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            cam_scale = 1000.0
            dpt_mm = zbuf * cam_scale
            dpt_mm = dpt_mm.copy().astype(np.uint16)
            # dpt_m = dpt_mm.astype(np.float32) / cam_scale
            # dpt_xyz = dpt_2_pcld(dpt_m, 1.0, self.K)
            # dpt_xyz[np.isnan(dpt_xyz)] = 0.0
            # dpt_xyz[np.isinf(dpt_xyz)] = 0.0
            nrm_map = normalSpeed.depth_normal(dpt_mm, self.K[0][0], self.K[1][1], 5, 2000, 20, False)
            # show_nrm_map = ((nrm_map + 1.0) * 127).astype(np.uint8)
            # cv2.imwrite('/workspace/DATA/Linemod_preprocessed/nrm_rn.png',show_nrm_map)
            #angle, signed = pseudo_gen(dpt_xyz)
            rgb_nrm = pseudo_nrm_angle(nrm_map)
            
            if args.vis:
                try:
                    from neupeak.utils.webcv2 import imshow, waitKey
                except ImportError:
                    from cv2 import imshow, waitKey
                imshow("bgr", bgr.astype("uint8"))
                show_zbuf = zbuf.copy()
                min_d, max_d = show_zbuf[show_zbuf > 0].min(), show_zbuf.max()
                show_zbuf[show_zbuf > 0] = (show_zbuf[show_zbuf > 0] - min_d) / (max_d - min_d) * 255
                show_zbuf = show_zbuf.astype(np.uint8)
                imshow("dpt", show_zbuf)
                show_msk = (msk / msk.max() * 255).astype("uint8")
                imshow("msk", show_msk)
                waitKey(0)
            
            sv_pth = os.path.join(self.render_dir, "{}".format(idx))
            np.savez_compressed(sv_pth, depth=zbuf, rgb=rgb, mask=msk, K=self.K, RT=RT, cls_typ=self.cls_type,rnd_typ='render', angles=rgb_nrm)
            pth_lst.append('/workspace/DATA/Linemod_preprocessed/renders_nrm/'+args.cls+'/'+str(idx)+'.npz')
            # data = {}
            # data['depth'] = zbuf
            # data['rgb'] = rgb
            # data['mask'] = msk
            # data['K'] = self.K
            # data['RT'] = RT
            # data['cls_typ'] = self.cls_type
            # data['rnd_typ'] = 'render'
            # sv_pth = os.path.join(self.render_dir, "{}.pkl".format(idx))
            # if DEBUG:
            #     imshow("rgb", rgb[:, :, ::-1].astype("uint8"))
            #     imshow("depth", (zbuf / zbuf.max() * 255).astype("uint8"))
            #     imshow("mask", (msk / msk.max() * 255).astype("uint8"))
            #     waitKey(0)
            # pkl.dump(data, open(sv_pth, "wb"))
            # pth_lst.append(os.path.abspath(sv_pth))

        plst_pth = os.path.join(self.render_dir, "file_list.txt")
        with open(plst_pth, 'w') as of:
            for pth in pth_lst:
                print(pth, file=of)


def main():
    cls_type = 'cam'
    render_num = 70000
    if len(args.cls) > 0:
        cls_type = args.cls
    if args.render_num > 0:
        render_num = args.render_num
    print("cls: ", cls_type)
    gen = LineModRenderDB(cls_type, render_num, True)
    gen.gen_pack_zbuf_render()

    # for cls_type in OBJ_ID_DICT.keys():
    #     pth_lst = []
    #     render_num = 70000
    #     gen = LineModRenderDB(cls_type, render_num, True)
    #     gen.gen_pack_zbuf_render()



if __name__ == "__main__":
    main()
# vim: ts=4 sw=4 sts=4 expandtab
