# modify from /projects/InternVideo/InternVideo2/multi_modality/tasks_clip/pretrain.py
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


logger = logging.getLogger(__name__)
world_size = 1  # 例如，你有1个进程
rank = 0  # 当前进程的排名，从0开始
# 设置环境变量
os.environ['RANK'] = str(rank)
os.environ['LOCAL_WORLD_SIZE'] = str(world_size)
os.environ['LOCAL_RANK'] = str(rank)  # 假设每个节点只有一个GPU

def main(config):

    if is_main_process() and config.wandb.enable:
        run = setup_wandb(config)

    is_pretrain = config.mode == "pt"

    logger.info(f"train_file: {config.train_file}")

    setup_seed(config.seed + get_rank())
    device = torch.device(config.device)

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


if __name__ == "__main__":

    cfg = setup_clip_main()
    local_broadcast_process_authkey()
    main(cfg)