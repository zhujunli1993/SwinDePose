from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import time
import tqdm
import resource
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import scheduler
import torch.backends.cudnn as cudnn


from config.options import BaseOptions
from config.common import Config, ConfigRandLA

import models.pytorch_utils as pt_utils
from models.SwinDePose import SwinDePose
from models.loss import OFLoss, FocalLoss
from utils.pvn3d_eval_utils_kpls import TorchEval
from utils.basic_utils import Basic_Utils
import datasets.lab.lab_dataset as dataset_desc



# from apex.parallel import convert_syncbn_model
# from apex import amp


    
# get options
opt = BaseOptions().parse()

config = Config(ds_name=opt.dataset_name, cls_type=opt.linemod_cls)
bs_utils = Basic_Utils(config)


rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (30000, rlimit[1]))

color_lst = [(0, 0, 0)]
for i in range(config.n_objects):
    col_mul = (255 * 255 * 255) // (i+1)
    color = (col_mul//(255*255), (col_mul//255) % 255, col_mul % 255)
    color_lst.append(color)




lr_clip = 1e-5
bnm_clip = 1e-2


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def checkpoint_state(model=None, optimizer=None, best_prec=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.DataParallel) or \
                isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {
        "epoch": epoch,
        "it": it,
        "best_prec": best_prec,
        "model_state": model_state,
        "optimizer_state": optim_state,
        "amp": amp.state_dict(),
    }


def save_checkpoint(
        state,  filename="checkpoint"
):
    filename = "{}.pth.tar".format(filename)
    torch.save(state, filename)



def load_checkpoint(model=None, optimizer=None, filename="checkpoint"):
    

    if os.path.isfile(filename):
        print("==> Loading from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        epoch = checkpoint["epoch"] 
        print("epoch: ", epoch)
        it = checkpoint.get("it", 0.0)
        best_prec = checkpoint["best_prec"]
        print("best_prec: ", best_prec)
        if model is not None and checkpoint["model_state"] is not None:
            ck_st = checkpoint['model_state']
            if 'module' in list(ck_st.keys())[0]:
                tmp_ck_st = {}
                for k, v in ck_st.items():
                    tmp_ck_st[k.replace("module.", "")] = v
                ck_st = tmp_ck_st
            model.load_state_dict(ck_st)
        if optimizer is not None and checkpoint["optimizer_state"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        # amp.load_state_dict(checkpoint["amp"])
        print("==> Done")
        return it, epoch, best_prec
    else:
        print("==> Checkpoint '{}' not found".format(filename))
        return None


def view_labels(rgb_chw, img_id, obj_id, cld_cn, labels, K=config.intrinsic_matrix['linemod']):
    
    #rgb_hwc = np.transpose(rgb_chw[0].numpy(), (1, 2, 0)).astype("uint8").copy()
    
    rgb_hwc = cv2.imread('/workspace/DATA/Linemod_preprocessed/data/'+str(obj_id)+'/rgb/'+img_id+'.png')
    cld_nc = np.transpose(cld_cn.numpy(), (1, 0)).copy()
    p2ds = bs_utils.project_p3d(cld_nc, 1.0, K).astype(np.int32)
    labels = labels.squeeze().contiguous().cpu().numpy()
    colors = []
    #h, w = rgb_hwc.shape[0], rgb_hwc.shape[1]
    #rgb_hwc = np.zeros((h, w, 3), "uint8")
    for lb in labels:
        if int(lb) == 0:
            c = (255, 255, 255)
        else:
            c = color_lst[int(lb)]
            #c = (0, 0, 0)
        colors.append(c)
    show = bs_utils.draw_p2ds(rgb_hwc, p2ds, 3, colors, 0.6)
    return show


def model_fn_decorator(
    criterion, criterion_of, test=False,
):
    teval = TorchEval()

    def model_fn(
        model, data, it=0, epoch=0, is_eval=False, is_test=False, finish_test=False,
        test_pose=False
    ):
        
        
        if is_eval:
            model.eval()
        with torch.set_grad_enabled(not is_eval):
            cu_dt = {}
            # device = torch.device('cuda:{}'.format(args.local_rank))
            for key in data.keys():
                if data[key].dtype in [np.float32, np.uint8]:
                    cu_dt[key] = torch.from_numpy(data[key].astype(np.float32)).cuda()
                elif data[key].dtype in [np.int32, np.uint32]:
                    cu_dt[key] = torch.LongTensor(data[key].astype(np.int32)).cuda()
                elif data[key].dtype in [torch.uint8, torch.float32]:
                    cu_dt[key] = data[key].float().cuda()
                elif data[key].dtype in [torch.int32, torch.int16]:
                    cu_dt[key] = data[key].long().cuda()
            
            start_time = time.time()
            
            end_points = model(cu_dt)
            print("--- %s seconds ---" % (time.time() - start_time))
            
            
            _, cls_rgbd = torch.max(end_points['pred_rgbd_segs'], 1)

            if is_test and test_pose:
                cld = cu_dt['cld_angle_nrm'][:, :3, :].permute(0, 2, 1).contiguous()
                
                if not opt.test_gt:
                    # eval pose from point cloud prediction.
                    pred_pose = teval.eval_pose_parallel_lab(
                        pclds = cld, masks = cls_rgbd, pred_ctr_ofs=end_points['pred_ctr_ofs'],
                        cnt=epoch, pred_kp_ofs=end_points['pred_kp_ofs'],
                        ds='lab', cls_ids=config.cls_id,
                        min_cnt=1, use_ctr_clus_flter=True, use_ctr=True
                    )
                
                
            
        return pred_pose

    return model_fn


class Trainer(object):
    """
        Reasonably generic trainer for pytorch models

    Parameters
    ----------
    model : pytorch model
        Model to be trained
    model_fn : function (model, inputs, labels) -> preds, loss, accuracy
    optimizer : torch.optim
        Optimizer for model
    checkpoint_name : str
        Name of file to save checkpoints to
    best_name : str
        Name of file to save best model to
    lr_scheduler : torch.optim.lr_scheduler
        Learning rate scheduler.  .step() will be called at the start of every epoch
    bnm_scheduler : BNMomentumScheduler
        Batchnorm momentum scheduler.  .step() will be called at the start of every epoch
    """

    def __init__(
        self,
        model,
        model_fn,
        optimizer,
        checkpoint_name="ckpt",
        lr_scheduler=None,
        bnm_scheduler=None,
        viz=None,
    ):
        self.model, self.model_fn, self.optimizer, self.lr_scheduler, self.bnm_scheduler = (
            model,
            model_fn,
            optimizer,
            lr_scheduler,
            bnm_scheduler,
        )

        self.checkpoint_name = checkpoint_name

        self.training_best, self.eval_best = {}, {}
        self.viz = viz

    def eval_epoch(self, d_loader, epoch,  is_test=False, test_pose=False):
        self.model.eval()

       
        
        eval_dict = {}
        total_loss = 0.0
        count = 1
        for _, data in enumerate(d_loader):
             
            count += 1
            self.optimizer.zero_grad()
            if opt.eval_net:
                
                pred_pose = self.model_fn(
                self.model, data, is_eval=True, is_test=is_test, test_pose=test_pose
            )
            else:
                pred_pose = self.model_fn(
                self.model, data, is_eval=True, is_test=is_test, test_pose=test_pose
            )
            
           
        return pred_pose
        

    def train(
        self,
        start_it,
        start_epoch,
        n_epochs,
        train_loader,
        train_sampler,
        test_loader=None,
        best_loss=0.0,
        log_epoch_f=None,
        tot_iter=1
    ):
        
        print("Totally train %d iters per gpu." % tot_iter)

        # Early stopping
        last_loss = 1e10
        patience = 7
        trigger_times = 0
        it = start_it
        for start_epoch in tqdm.tqdm(range(n_epochs)):
            
            if train_sampler is not None:
                train_sampler.set_epoch(start_epoch)
            # Reset numpy seed.
            # REF: https://github.com/pytorch/pytorch/issues/5059
            np.random.seed()
            if log_epoch_f is not None:
                os.system("echo {} > {}".format(start_epoch, log_epoch_f))
            
            
            
            for batch in tqdm.tqdm(train_loader):

                
                self.model.train()

                self.optimizer.zero_grad()
                _, loss, res = self.model_fn(self.model, batch, it=it)

                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()

                self.optimizer.step()

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step(it)

                if self.bnm_scheduler is not None:
                    self.bnm_scheduler.step(it)

                it += 1
                
                if self.viz is not None:
                    self.viz.update("train", it, res)

                
                
            
            if test_loader is not None:
                if opt.eval_net:
                    val_loss, res, _ = self.eval_epoch(test_loader, start_epoch)
                else:
                    val_loss, res = self.eval_epoch(test_loader, start_epoch)
                if val_loss < best_loss:
                    best_loss = val_loss
                    if opt.local_rank == 0:
                        save_checkpoint(
                            checkpoint_state(
                                self.model, self.optimizer, val_loss, start_epoch, it
                            ),
                            filename=self.checkpoint_name)
                # Early Stopping
                current_loss = val_loss
                if current_loss > last_loss:
                    trigger_times += 1
                    if trigger_times >= patience:
                        print('Early Stopping!\n')
                        exit()
                else:
                    trigger_times = 0
                last_loss = current_loss
        
        return val_loss


def train():
    
    
    print("local_rank:", opt.local_rank)
    cudnn.benchmark = True
    if opt.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(opt.local_rank)
        torch.set_printoptions(precision=10)
    torch.cuda.set_device(opt.local_rank)
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
    )

    test_ds = dataset_desc.Dataset('test', cls_type=opt.linemod_cls)
    test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=opt.test_mini_batch_size, shuffle=False,
            num_workers=opt.num_threads
        )

    rndla_cfg = ConfigRandLA
    model = SwinDePose(
        n_classes=config.n_objects, n_pts=opt.n_sample_points, rndla_cfg=rndla_cfg,
        n_kps=opt.n_keypoints
    )

    print("Number of model parameters: ", count_parameters(model))
    # model = convert_syncbn_model(model)
    device = torch.device('cuda:{}'.format(opt.local_rank))
    print('local_rank:', opt.local_rank)
    model.to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay
    )
    opt_level = opt.opt_level
    # model, optimizer = amp.initialize(
    #     model, optimizer, opt_level=opt_level,
    # )

    # default value
    it = -1  # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
    best_loss = 1e10
    start_epoch = 1

    # load status from checkpoint
    if opt.load_checkpoint is not None:
        
        checkpoint_status = load_checkpoint(
            model, optimizer, filename=opt.load_checkpoint
        )
        if checkpoint_status is not None:
            it, start_epoch, best_loss = checkpoint_status
        if opt.eval_net:
            assert checkpoint_status is not None, "Failed loadding model."

    lr_scheduler = None

    bnm_lmbd = lambda it: max(
        opt.bn_momentum * opt.bn_decay ** (int(it * opt.mini_batch_size / opt.decay_step)),
        bnm_clip,
    )
    bnm_scheduler = pt_utils.BNMomentumScheduler(
        model, bn_lambda=bnm_lmbd, last_epoch=it
    )

    it = max(it, 0)  # for the initialize value of `trainer.train`

    if opt.eval_net:
        model_fn = model_fn_decorator(
            FocalLoss(gamma=2), OFLoss(),
            opt.test, 
        )
    else:
        model_fn = model_fn_decorator(
            FocalLoss(gamma=2).to(device), OFLoss().to(device),
            opt.test,
        )

    checkpoint_fd = opt.save_checkpoint

    trainer = Trainer(
        model,
        model_fn,
        optimizer,
        checkpoint_name=checkpoint_fd,
        lr_scheduler=lr_scheduler,
        bnm_scheduler=bnm_scheduler,
    )

    if opt.eval_net:
        
        pred_pose = trainer.eval_epoch(
            test_loader, opt.n_total_epoch, is_test=True, test_pose=opt.test_pose
        )
        print("-------------------Estimation Result---------------------------\n")
        print(pred_pose)
        np.savetxt(os.path.join(opt.lab_pose_save, opt.linemod_cls+'_pred.txt'),pred_pose)
        
        # save test results
        
        
        

if __name__ == "__main__":
    opt.world_size = opt.gpus * opt.nodes
    
    train()
