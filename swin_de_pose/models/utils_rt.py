import json
import os
import random
import warnings

# import tensorflow as tf
import torch
import torch.optim as optim
# from loguru import logger
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms.transforms import Compose, Lambda

# Ignore warnings
warnings.filterwarnings("ignore")


def adjust_learning_rate(optimizer, multiplier, stop_lr):
    """Sets the learning rate to the initial LR decayed by multiplier when
    called
    :param optimizer: optimizer used
    :param multiplier: LR is decreased by this factor
    """
    for param_group in optimizer.param_groups:
        newlr = param_group["lr"] * multiplier
        if newlr > (stop_lr):
            param_group["lr"] = newlr


def restore_from_file(args,
                      model,
                      optimizer,
                      rank=0,
                      model_dir=os.path.join(".", "model")):
    """
    Loads weights from single frame prediction model, which is a different
    model from the current model.
    Takes in complete path.
    :param args: Args
    :param model: Model to load to
    :return:
    """
    # Checks if the restore path is a file, or a full path to the folder.
    if os.path.isfile(args.restore_file):
        checkpoint_path = args.restore_file
    else:
        if os.path.isabs(args.restore_file):
            restore_path = args.restore_file
        else:
            restore_path = os.path.join(model_dir, args.restore_file)

        counter = args.start_epoch
        # If restoring from the final epoch, do not have to specify the epoch
        if args.start_epoch == 0:
            counter = int(
                sorted(
                    os.listdir(restore_path))[-1].split("_")[-1].split(".")[0])

        checkpoint_path = os.path.join(
            restore_path, "last_checkpoint_{}.tar".format("%04d" % counter))

    checkpoint = load_checkpoint(checkpoint_path)
    epoch_check = checkpoint["epoch"]
    avg_loss = 0
    if "accuracy" in checkpoint.keys():
        avg_loss = checkpoint["accuracy"]

    is_partial = 0
    if "is_partial" in checkpoint.keys():
        is_partial = checkpoint["is_partial"]

    model.load_state_dict(checkpoint["state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optim_dict"])

    # sanity check that copy is done correctly.
    for name, param in model.named_parameters():
        keys = checkpoint["state_dict"].keys()
        if name in keys:
            assert torch.allclose(param.data, checkpoint["state_dict"][name])

    return model, optimizer, epoch_check, is_partial


# def load_from_tensorflow(model):
#     # tensorflow part
#     posecnn = getattr(model, "backbone")
#     tf_path = os.path.join(
#         ".",
#         "model",
#         "posecnn",
#         "vgg16_fcn_color_single_frame_2d_pose_add_lov_iter_160000.ckpt",
#     )
#     init_vars = tf.train.list_variables(tf_path)
#     tf_vars = []
#     for name, shape in init_vars:
#         array = tf.train.load_variable(tf_path, name)
#         tf_vars.append((name, array.squeeze()))

#     for name, array in tf_vars:
#         name = name.split("/")
#         if name[0] == "Variable":
#             continue
#         if len(name) < 3:
#             if name[1] == "biases":
#                 name[1] = "bias"
#             if name[1] == "weights":
#                 name[1] = "weight"
#             layer = None
#             try:
#                 layer = getattr(posecnn, name[0])
#             except:
#                 pass
#             # try:
#             #     if name[0] != "vertex_pred":
#             #         layer = getattr(depth, name[0])
#             # except:
#             #     pass

#             if layer is not None:
#                 if len(list(layer.named_children())) > 0:
#                     conv = getattr(layer[0], name[1])
#                 else:  # Non Sequential
#                     conv = getattr(layer, name[1])

#                 if len(array.shape) == 4:  # conv2d
#                     conv.data = torch.from_numpy(array).permute(3, 2, 0,
#                                                                 1).float()
#                 elif len(array.shape) == 2 and len(conv.data.shape) == 4:
#                     conv.data[:, :, 0,
#                               0] = (torch.from_numpy(array).permute(1,
#                                                                     0).float())
#                 elif len(array.shape) == 2 and len(conv.data.shape) == 2:
#                     conv.data = torch.from_numpy(array).permute(1, 0).float()
#                 else:
#                     conv.data = torch.from_numpy(array).float()
#     return model


def get_optimizer(args, optim_params):
    """
    Initialises and returns an optimizer
    :param args:
    :param optim_params: model optim parameters
    :return: optimizer
    """
    if args.optimiser == "adam":
        optimizer = optim.Adam(optim_params,
                               lr=args.lr,
                               eps=1e-4,
                               weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(
            optim_params,
            lr=args.lr,
            nesterov=True,
            weight_decay=args.weight_decay,
            momentum=0.9,
            dampening=0,
        )
    return optimizer


class RunningAverage:
    """A simple class that maintains the running average of a quantity
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps) if self.steps != 0 else 0


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, "w") as f:
        # We need to convert the values to float for json
        # (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


# def save_checkpoint(state, checkpoint, counter=0):
#     """Saves model and training parameters at checkpoint + 'last.pth.tar'.
#     If is_best==True, also saves
#     checkpoint + 'best.pth.tar'
#     Args:
#         state: (dict) contains model's state_dict, may contain other keys such
#         as epoch, optimizer state_dict
#         is_best: (bool) True if it is the best model seen till now
#         checkpoint: (string) folder where parameters are to be saved
#     """
#     filepath = os.path.join(checkpoint,
#                             "last_checkpoint_{}.tar".format(counter))
#     if not os.path.exists(checkpoint):
#         logger.info(
#             "Checkpoint Directory does not exist! Making directory {}".format(
#                 checkpoint))
#         os.mkdir(checkpoint)
#     else:
#         logger.info("Checkpoint Directory exists! ")
#     torch.save(state, filepath)


# def load_checkpoint(checkpoint_path):
#     """Loads model parameters (state_dict) from file_path.
#     If optimizer is provided, loads state_dict of
#     optimizer assuming it is present in checkpoint.
#     Args:
#         checkpoint_path: (string) filename which needs to be loaded
#     """

#     if not os.path.exists(checkpoint_path):
#         raise ValueError("File doesn't exist: {}".format(checkpoint_path))

#     logger.info("Restoring parameters from {}".format(checkpoint_path))
#     checkpoint = torch.load(checkpoint_path)

#     return checkpoint


class Transform(object):
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

    def __call__(self, add_jitter=False):
        if add_jitter:
            transform = transforms.Compose(
                [self.colorJitter(),
                 self.to_tensor(),
                 self.normalise()])
        else:
            transform = transforms.Compose(
                [self.to_tensor(), self.normalise()])
        return transform

    def to_tensor(self):
        return transforms.ToTensor()

    def normalise(self):
        return transforms.Normalize(self.mean.tolist(), self.std.tolist())

    def unnormalise(self):
        return transforms.Normalize((-self.mean / self.std).tolist(),
                                    (1.0 / self.std).tolist())

    def colorJitter(self):
        return transforms.ColorJitter(0.4, 0.2, 0.2, 0.1)


class ColorJitter(object):
    """
    Redefinition of ColorJitter class. This gives us more control over the
    jitter when running over a video sequence compared to torchvision which
    applies a different value for each sequence. We want to have consistent
    jitter across the video sequence.
    """
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

        self.brightness_factor = None
        self.contrast_factor = None
        self.saturation_factor = None
        self.hue_factor = None

    def __call__(self, img):
        transform = self.get_params()
        return transform(img)

    def set_factors(self):
        assert self.hue < 0.5
        self.brightness_factor = random.uniform(1 - self.brightness,
                                                1 + self.brightness)
        self.contrast_factor = random.uniform(1 - self.contrast,
                                              1 + self.contrast)
        self.saturation_factor = random.uniform(1 - self.saturation,
                                                1 + self.saturation)
        self.hue_factor = random.uniform(-self.hue, self.hue)

    def get_params(self):
        transforms = []
        transforms.append(
            Lambda(
                lambda img: F.adjust_brightness(img, self.brightness_factor)))
        transforms.append(
            Lambda(lambda img: F.adjust_contrast(img, self.contrast_factor)))
        transforms.append(
            Lambda(
                lambda img: F.adjust_saturation(img, self.saturation_factor)))
        transforms.append(
            Lambda(lambda img: F.adjust_hue(img, self.hue_factor)))
        random.shuffle(transforms)
        transforms = Compose(transforms)
        return transforms


class WriteLogs:
    def __init__(self, writer=None, wandb=None):
        self.writer = writer
        self.wandb = wandb

    def log_scalars_dict(self, key, val, step=0):
        if self.writer is not None:
            self.writer.add_scalars(key, val, step)
        if self.wandb is not None:
            self.wandb.log({key: val})

    def log_scalar(self, key, val, step=0):
        if self.writer is not None:
            self.writer.add_scalar(key, val, step)
        if self.wandb is not None:
            self.wandb.log({key: val})

    def log_images(self, key, image, step=0):
        if self.writer is not None:
            self.writer.add_image(key, image, global_step=0)
        if self.wandb is not None:
            if len(image.shape) > 2:
                image = image.transpose(1, 2, 0)
            self.wandb.log({key: [self.wandb.Image(image, caption=key)]})

    def log_text(self, key, text, step=0):
        if self.writer is not None:
            self.writer.add_text(key, text)
        if self.wandb is not None:
            self.wandb.log({key: text})