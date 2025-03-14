import os

from os.path import join as ospj
from os.path import expanduser
from munch import Munch as mch
import numpy as np

from ds import prepare_scannet_test_dataloaders, prepare_scannet_scene_dataloaders

from utils import *
# from networks import *
from test_ProbVLM import *
from clip_adapter import *
import matplotlib.pyplot as plt
from time import gmtime, strftime

dataset = 'scannetpp' # coco or flickr
data_dir = ospj('../../Datasets/', dataset) # e.g. ospj(expanduser('~'), 'Documents', 'jm', 'data', dataset)
dataloader_config = mch({
    'batch_size': 64,
    'random_erasing_prob': 0.,
    'traindata_shuffle': True,
    'loaders': ['train', 'val', 'test']
})

######### TRAIN THE CLIP ADAPTER ##########

for pt in [0.26, 0.28, 0.30, 0.32]:
    # for r in [0.1, 0.2, 0.3, 0.4, 0.5]:
    for r in [0.3]:
        dataset_cfg = mch({
            'pseudo_label': 'caption',
            'pseudo_thresh': pt,
            'seen_classes': 'top100_half',
            'pseudo_method': 'scene',
            'use_affordances': True,
        })

        loaders,vocab = prepare_scannet_scene_dataloaders(dataloader_config, dataset_root=data_dir, caption_root=data_dir+'/data_download/complete_dataset/finetune_data/dataset01', cfg = dataset_cfg)

        cub_train_loader, cub_valid_loader, cub_test_loader = loaders['train'], loaders['val'], loaders['test']

        CLIP_Net = load_model(device='cuda', model_path=None)

        adapter_cfg = mch({
            'ratio': r,
            'mean_score': 0.2,
            'logit_scale': np.log(50),
        })

        train_clip_adapter(
            CLIP_Net,
            cub_train_loader,
            cub_test_loader,
            device='cuda',
            lr=1e-3,
            dtype=torch.cuda.FloatTensor,
            num_epochs=500,
            eval_every=10,
            ckpt_path=f'../ckpt/08_Caption_Grid_Search/ratio{r}_thresh{pt}/',
            cfg=adapter_cfg
        )
