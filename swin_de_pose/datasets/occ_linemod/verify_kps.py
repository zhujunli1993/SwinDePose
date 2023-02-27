import numpy as np
import cv2

K=np.array([[572.4114, 0.,         325.2611],
            [0.,        573.57043,  242.04899],
            [0.,        0.,         1.]])
def draw_p2ds(img, p2ds, r=1, color=[(255, 0, 0)],alpha=1):
    # alpha = 0.4  # Transparency factor.
    # pick every 10 points
    # p2ds = p2ds[::10, :]
    overlay = img.copy()
    if type(color) == tuple:
        color = [color]
    if len(color) != p2ds.shape[0]:
        color = [color[0] for i in range(p2ds.shape[0])]
    h, w = img.shape[0], img.shape[1]
    for pt_2d, c in zip(p2ds, color):

        pt_2d[0] = np.clip(pt_2d[0], 0, w)
        pt_2d[1] = np.clip(pt_2d[1], 0, h)
        # img = cv2.circle(
        #     img, (pt_2d[0], pt_2d[1]), r, c, -1
        # )
        cv2.circle(
            overlay, (pt_2d[0], pt_2d[1]), r, c, -1
        )
        # alpha = 0.4  # Transparency factor.
        # # Following line overlays transparent rectangle over the image
        # img = cv2.addWeighted(img, alpha, img, 1 - alpha, 0)
    image_new = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    #return img
    return image_new
def project_p3d( p3d, cam_scale, K=K):

    p3d = p3d * cam_scale
    p2d = np.dot(p3d, K.T)
    p2d_3 = p2d[:, 2]
    p2d_3[np.where(p2d_3 < 1e-8)] = 1.0
    p2d[:, 2] = p2d_3
    p2d = np.around((p2d[:, :2] / p2d[:, 2:])).astype(np.int32)
    return p2d
def draw_points(pred_RT, p3ds):
    
    pred_p3ds = np.dot(p3ds, pred_RT[:, :3].transpose(1, 0)) + pred_RT[:, 3]
    # for cropping image
    # show_kp_img = cv2.imread('/workspace/DATA/Linemod_preprocessed/data/'+str(obj_id)+'/crop_rgb/'+img_id+'.png')
    show_kp_img = cv2.imread('/workspace/DATA/Occ_LineMod/train_pbr/000000/mask/000000_000006.png')
    pred_2ds = project_p3d(
        pred_p3ds, 1000.0, K=K
    )

    color = (0, 0, 255)  # bs_utils.get_label_color(cls_id.item())
    show_kp_img = draw_p2ds(show_kp_img, pred_2ds, r=3, color=color)
    # imshow("kp: cls_id=%d" % cls_id, show_kp_img)
    cv2.imwrite('/workspace/DATA/Occ_LineMod/train_pbr/000000_kps.png', show_kp_img)

p3ds = np.loadtxt('/workspace/DATA/Occ_LineMod/kps_orb9_fps/ape_8_kps.txt')
t = [0.22545530700683594, 0.16851051330566406, 1.008190185546875]
R = [[-0.01795329339802265, -0.959463894367218, 0.28125935792922974], [-0.1360679417848587, -0.2763430178165436, -0.9513779878616333], [0.9905368685722351, -0.0553507208824, -0.12559105455875397]]

pred_RT = np.zeros((3,4))
pred_RT[:,0:3] = R
pred_RT[:,3] = t
draw_points(pred_RT, p3ds)