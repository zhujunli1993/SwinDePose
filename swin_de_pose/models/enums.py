from enum import Enum
from typing import Dict, TypedDict

import numpy as np
import torch

import utils_rt as util

float_pt = torch.FloatTensor


class TrainingSample(Dict):
    image: torch.Tensor = torch.Tensor()
    poses: torch.Tensor = torch.Tensor()
    cls_indices: torch.Tensor = torch.Tensor()
    intrinsic: torch.Tensor = torch.Tensor()
    extrinsic: torch.Tensor = torch.Tensor()
    bbox: torch.Tensor = torch.Tensor()
    label: torch.Tensor = torch.Tensor()
    depth: torch.Tensor = torch.Tensor()
    posecnn_poses: torch.Tensor = torch.Tensor()
    posecnn_bbox: torch.Tensor = torch.Tensor()
    prev_state: torch.Tensor = torch.Tensor()
    future_feat: torch.Tensor = torch.Tensor()
    is_keyframe: torch.Tensor = torch.Tensor()
    file_indices: list = []


class Output(TypedDict):
    _R: torch.Tensor
    _T: torch.Tensor
    pose_out: torch.Tensor
    output_losses: dict
    loss_value: float
    _depth: np.ndarray
    _label: np.ndarray
    prev_state: torch.Tensor


class LossDict():
    def __init__(self):

        self.loss_avg = util.RunningAverage()
        self.rotation_distance_loss = util.RunningAverage()
        self.rt_loss = util.RunningAverage()

    def __call__(self):
        return self.loss_avg()

    def get_rot_loss(self):
        return self.rotation_distance_loss()

    def get_rt_loss(self):
        return self.rt_loss()

    def get_loss_avg(self):
        return self.__call__()


class Losses(TypedDict):
    train: LossDict = LossDict()
    test: LossDict = LossDict()
    eval: LossDict = LossDict()


class Split(Enum):
    Train = "train"
    Test = "test"
    Evaluate = "eval"