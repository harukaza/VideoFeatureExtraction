# modify from demo_video_text_retrieval.ipynb
# 利用 internvideo2_stage2 提取特征
import numpy as np
import os
import io
import cv2

import torch

from extract_config import (Config,
                    eval_dict_leaf)
from extract_utils import (frames2tensor,
                  _frame_from_video,
                  setup_internvideo2)

def load_video():
    video = cv2.VideoCapture('/projects/InternVideo/InternVideo2/multi_modality/demo/example1.mp4')
    frames = [x for x in _frame_from_video(video)]
    return frames

def retrieve_text(frames,
                  model,
                  topk:int=5,
                  config: dict={},
                  device=torch.device('cuda')):

    vlm = model
    vlm = vlm.to(device)

    fn = config.get('num_frames', 8)
    size_t = config.get('size_t', 224)
    frames_tensor = frames2tensor(frames, fnum=fn, target_size=(size_t, size_t), device=device)
    vid_feat = vlm.get_vid_feat(frames_tensor) # feature shapes: [1, 512]
    vid_feat2 = vlm.encode_vision(frames_tensor)

    return None

def main():
    config = Config.from_file('/projects/InternVideo/InternVideo2/multi_modality/video_extract/config.py')
    config = eval_dict_leaf(config)
    intern_model, tokenizer = setup_internvideo2(config)
    frames = load_video()
    retrieve_text(frames, model=intern_model, topk=5, config=config)


if __name__ == "__main__":
    main()