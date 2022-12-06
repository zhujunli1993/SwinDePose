import torch
import torch.nn.functional as F
# from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
from torch import nn

from enums import float_pt

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret
def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))
def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))




class Loss(nn.Module):

    def __init__(self, losses, bs_utils, config, args):
        super().__init__()
        # self.sym_objects = [13, 16, 19, 20, 21]
        self.sym_objects = []
        self.losses = losses
        self.args = args

        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")
        self.loss_l2 = torch.nn.MSELoss(reduction="none")
        self.loss_l2_rt = torch.nn.MSELoss(reduction="none")
        self.loss_l1 = torch.nn.L1Loss(reduction="none")
        self.loss_l1_depth = torch.nn.L1Loss(reduction="none")

        self.beta = 1
        self.gamma = 1
        # root = args.data_root_path
        
        # self.points = torch.from_numpy(points)

    @staticmethod
    def normalize_quaternion(quat, eps=1e-12):
        r"""Normalizes a quaternion.
        The quaternion should be in (w, x, y, z) format.
        Args:
            quaternion (torch.Tensor): a tensor containing a quaternion
            to be
              normalized. The tensor can be of shape *, 4.
            eps (Optional[bool]): small value to avoid division by zero.
              Default: 1e-12.
        Return:
            torch.Tensor: the normalized quaternion of shape *, 4.
        """
        if not isinstance(quat, torch.Tensor):
            raise TypeError("Input type is not a torch.Tensor. Got {}".format(
                type(quat)))

        if not quat.shape[-1] == 4:
            raise ValueError(
                "Input must be a tensor of shape (*, 4). Got {}".format(
                    quat.shape))

        return F.normalize(quat, p=2, dim=-1, eps=eps)

    def flow_loss(self, input_flow, target_flow):
        EPE = torch.norm(target_flow - input_flow, p=2, dim=1).mean()
        l1 = self.loss_l1(target_flow, input_flow).mean()
        return (l1 + EPE).mean()

    def quat_loss(self, output_r, poses, eps=1e-7):
        
        rot = poses[:, :, 0:3].contiguous()
        gt = matrix_to_quaternion(rot)

        est = self.normalize_quaternion(output_r)
        inn_prod = torch.mm(est, gt.t())
        inn_prod = inn_prod.diag()

        quat_loss = 1 - (inn_prod).abs().mean()
        quat_regularisation_loss = (1 - output_r.norm(dim=1).mean()).abs()

        return quat_loss, quat_regularisation_loss

    @staticmethod
    def rotate_pts(R, points):
        Rp = R.bmm(points.transpose(1, 2))
        return Rp

    @staticmethod
    def rotate_translate_pts(R, t, points):
        Rp = R.bmm(points.transpose(1, 2))
        t_ = t.unsqueeze(2).repeat(1, 1, Rp.size(2))
        return Rp.add(t_)

    @staticmethod
    def transfrom_pts_2d(K, _3d_points):
        points_2d = K.bmm(_3d_points.transpose(1, 2))
        points_2d = points_2d.transpose(1, 2)
        points_2d[:, :, 0] = torch.clamp(torch.div(points_2d[:, :, 0],
                                                   points_2d[:, :, 2]),
                                         min=0,
                                         max=639)
        points_2d[:, :, 1] = torch.clamp(torch.div(points_2d[:, :, 1],
                                                   points_2d[:, :, 2]),
                                         min=0,
                                         max=479)
        return points_2d[:, :, 0:2].int()

    def _3d_distance_loss(self, output_r, output_t, poses, pts):
        
        output_q = output_r.view(-1, 4)
        rot_mat = quaternion_to_matrix(output_q)

        # reshape the various rotation and translation matrices
        # pts = pts.flatten(0, 1)
        R_gt = poses[:, :, 0:3].float()
        t_gt = poses[:, :, 3].float()
        t_est = output_t.float()
        # 3D_loss is L1. Maintaining consistency.
        trans_loss = torch.mul(
            torch.pow(t_est.sub(t_gt), 2).abs().sum(dim=1).sqrt().mean(), 100)

        gt_temp = self.rotate_pts(R_gt, pts)
        est_temp = self.rotate_pts(rot_mat, pts)

        gt_3d = gt_temp.permute((0, 2, 1))
        est_3d = est_temp.permute((0, 2, 1))
        _3d_loss = (torch.mul(self.loss_l2(gt_3d, est_3d),
                              10000).sum(dim=2).sqrt().mean())

        gt_temp_rt = self.rotate_translate_pts(R_gt, t_gt, pts)
        est_temp_rt = self.rotate_translate_pts(rot_mat, t_est, pts)

        gt_3d_rt = gt_temp_rt.permute((0, 2, 1))
        est_3d_rt = est_temp_rt.permute((0, 2, 1))
        rt_loss = (torch.mul(self.loss_l2(gt_3d_rt, est_3d_rt),
                             10000).sum(dim=2).sqrt().mean())

        return trans_loss, _3d_loss, rt_loss

    def iou_loss(
            self,
            x,
            output_r,
            output_t,
            amodal,
            K,
            pts,
            cls_indices,
            kernel=torch.ones((1, 1, 5, 5)),
    ):
        output_q = output_r.view(-1, 4)
        rot_mat = quaternion_to_matrix(output_q)

        # reshape the various rotation and translation matrices
        t_est = output_t.view(-1, 3).float()

        est_temp = self.rotate_translate_pts(rot_mat, t_est, pts)
        est_3d = est_temp.permute((0, 2, 1))
        est_2d = self.transfrom_pts_2d(K, est_3d)

        amodal_est = torch.zeros((est_temp.shape[0], x.shape[2], x.shape[3]))
        amodal_est[:, est_2d[:, :, 1].long(),
                   est_2d[:, :, 0].long()] = float_pt([1]).type_as(x)
        amodal_est = amodal_est.type_as(x).requires_grad_()
        _b, _c, _h, _w = amodal.shape
        amodal_gt = amodal.contiguous().view(-1, 1, _h, _w)
        amodal_gt = (amodal_gt > 0) * 1
        amodal_dilate = (F.conv2d(
            amodal_est.unsqueeze(1), kernel.type_as(x), padding=2) > 5) * 1
        total_pixels = amodal_gt.sum()
        loss = 1 - (amodal_gt.float() *
                    amodal_dilate.float()).sum() / (total_pixels)
        return loss.type_as(x).requires_grad_()

    def iou_loss_roi(self, labels_gt, labels_est):
        union = ((labels_est + labels_gt) > 0).sum()
        intersection = (labels_gt * labels_est).sum()

        num_pixels = labels_est.shape[0] * labels_est.shape[
            2] * labels_est.shape[3]

        reg_penalty = union / num_pixels
        return (1 - intersection / union) + reg_penalty

    def forward(
        self,
        output_r,
        output_t,
        poses,
        K,
        cls_indices,
        points,
        losses
    ):
        
        poses = poses.type_as(K)
        # poses = poses.type_as(x)  # batchXnum_boxesX3X4
        # K = K.type_as(x).float()

        if len(output_r.shape) > 3:
            output_r = output_r.reshape(-1, 4)
            output_t = output_t.reshape(-1, 3)
            cls_indices = cls_indices.reshape(-1)
            poses = poses.reshape(-1, 3, 4)

        # 3d loss
        ################
        # This should be fast since not a lot of heavy operations here.
        # pts = []
        # for j in range(0, output_r.size(0)):  # batch size
        #     pts.append(self.points[cls_indices[j].type(torch.LongTensor) - 1])
        # pts = torch.stack(pts).type_as(output_r)

        # non-zero indices - To maintain same number of boxes, data is appendded with zeros. See dataloader.py collate_fn
        # ind = torch.nonzero(poses.sum(1).sum(1))[:, 0]
        # pts = self.points[ind]
        # output_r = output_r[ind]
        # output_t = output_t[ind]
        # poses = poses[ind]

        poses = poses[:,0,:,:]
        
        # quat loss should be calculated for only non symmetrical objects.
        quat_loss, quat_regularisation_loss = self.quat_loss(output_r, poses)
        translation_loss, r_loss, rt_loss = self._3d_distance_loss(
            output_r, output_t, poses, points)
        
        loss_ = float_pt([0]).type_as(K).requires_grad_()

        loss_dict = {}
        loss_dict["Quat_loss"] = quat_loss.reshape(1)
        loss_dict["rotation_distance_loss"] = r_loss.reshape(1)
        loss_dict["translation_loss"] = translation_loss.reshape(1)
        loss_dict["rt_loss"] = rt_loss.reshape(1)
        loss_dict["quat_norm_loss"] = quat_regularisation_loss.reshape(1)

        if "quat" in self.losses:
            loss_ = loss_ + quat_loss
        if "ADD" in self.losses:
            loss_ = loss_ + torch.mul(rt_loss, self.beta)
            loss_ = loss_ + quat_regularisation_loss
        if "rot" in self.losses:
            loss_ = torch.mul(r_loss, self.gamma) + loss_
        if "trans" in self.losses:
            loss_ = loss_ + torch.mul(translation_loss, self.gamma)
        if "ADD_sep" in self.losses:
            loss_ = (loss_ + torch.mul(r_loss, self.beta) +
                     torch.mul(translation_loss, self.gamma))
            loss_ = loss_ + quat_regularisation_loss

        
        loss_ = loss_.reshape(-1)
        return loss_, loss_dict