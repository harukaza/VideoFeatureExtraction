# modify from /InternVideo/InternVideo2/multi_modality/tasks_clip/pretrain.py
import datetime
import logging
import time
from os.path import join

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import wandb
from torch.utils.data import ConcatDataset

import os
import sys
sys.path.append('/projects/InternVideo/InternVideo2/multi_modality')

from dataset.serialize import local_broadcast_process_authkey
from dataset import MetaLoader_rs, create_dataset, create_loader, create_sampler, create_stateful_sampler
from models import *
from tasks_clip.retrieval_utils import evaluation_wrapper
from tasks_clip.shared_utils import get_media_types, setup_model
from utils.basic_utils import MetricLogger, SmoothedValue, setup_seed
from utils.config_utils import setup_clip_main
from utils.distributed import get_rank, is_main_process
from utils.logger import log_dict_to_wandb, setup_wandb

from clip_loader import get_video_loader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import numpy as np

logger = logging.getLogger(__name__)
world_size = 1  # 例如，你有1个进程
rank = 0  # 当前进程的排名，从0开始
# 设置环境变量
os.environ['RANK'] = str(rank)
os.environ['LOCAL_WORLD_SIZE'] = str(world_size)
os.environ['LOCAL_RANK'] = str(rank)  # 假设每个节点只有一个GPU

def test_transform_init():
    # from /InternVideo/InternVideo2/multi_modality/dataset/__init__.py L133-154
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    normalize = transforms.Normalize(mean, std)

    # loaded images and videos are torch.Tensor of torch.uint8 format,
    # ordered as (T, 1 or 3, H, W) where T=1 for image   , # NxHxWx3, (16, 480, 640, 3)
    type_transform = transforms.Lambda(lambda x: x.float().div(255.0))

    test_transform = transforms.Compose(
        [
            transforms.Resize(
                (224, 224),
                interpolation=InterpolationMode.BICUBIC,
            ),
            type_transform,
            normalize,
        ]
    )
    return test_transform

def main(config):

    # get model
    if is_main_process() and config.wandb.enable:
        run = setup_wandb(config)

    is_pretrain = config.mode == "pt"

    logger.info(f"train_file: {config.train_file}")

    setup_seed(config.seed + get_rank())
    device = torch.device(config.device)


    config.scheduler.num_training_steps = 1 * config.scheduler.epochs
    config.scheduler.num_warmup_steps = 1 * config.scheduler.warmup_epochs

    model_cls = eval(config.model.get('model_cls', 'InternVideo2_CLIP'))
    (
        model,
        model_without_ddp,
        optimizer,
        scheduler,
        scaler,
        tokenizer,
        start_epoch,
        global_step,
    ) = setup_model(
        config,
        model_cls=model_cls,
        pretrain=is_pretrain,
        find_unused_parameters=True,
    )
    if is_main_process() and config.wandb.enable:
        wandb.watch(model)

    if config.get('use_bf16', True):
        data_type = torch.bfloat16
    else:
        data_type = torch.float16

    # prepare data
    video_loader = get_video_loader()
    feature_list = []
    transform=test_transform_init()

    video_path = "/InternVideo/InternVideo2/multi_modality/demo/example1.mp4"
    vr = video_loader(video_path)
    data = vr.get_batch(np.arange(0, 0 + 16)).numpy() # NxHxWx3, (16, 480, 640, 3)
    data = np.transpose(data, (0, 3, 1, 2)) # (T, 1 or 3, H, W) where T=1 for image
    frame = torch.from_numpy(data)
    frame_q = transform(frame)
    input_data = frame_q.unsqueeze(0).cuda()
    print(input_data.shape)
    with torch.no_grad():
        feature = model.encode_vision(input_data)   # [B,T,C,H,W] -> [B,C,T,H,W], out [1, 768]
        print(feature.shape)
        feature_list.append(feature.float().cpu().numpy())


    if is_main_process() and config.wandb.enable:
        run.finish()

    # np.save(url, np.vstack(feature_list))
    # print(f'[{idx} / {num_videos}]: save feature on {url}')
    print("done")




if __name__ == "__main__":

    cfg = setup_clip_main()
    local_broadcast_process_authkey()
    main(cfg)
