import sys
sys.path.append('../')
import os

from os.path import join as ospj
from os.path import expanduser
from munch import Munch as mch
import numpy as np

from ds import prepare_scannet_test_dataloaders, prepare_scannet_scene_dataloaders, prepare_conceptgraph_dataloaders

from utils import *
# from networks import *
from test_ProbVLM import *
from clip_adapter import *
import matplotlib.pyplot as plt
from time import gmtime, strftime
import open_clip

dataset = 'scannetpp' # coco or flickr
data_dir = ospj('../../../Datasets/', dataset) # e.g. ospj(expanduser('~'), 'Documents', 'jm', 'data', dataset)
dataloader_config = mch({
    'batch_size': 64,
    'random_erasing_prob': 0.,
    'traindata_shuffle': True,
    'loaders': ['train', 'val', 'test']
})

######### TRAIN THE CLIP ADAPTER ##########

# TODO - add the proportion of classes used for training, and sweep it with and without pseudo-labels
# TODO - sweep pseudo_thresh
# TODO - review pseudo-labeling process (eg. pseudo-labelling happens in response to a query)

if True:
    dataset_cfg = mch({
        'pseudo_label': 'continual',
        'pseudo_thresh': 0.32,
        'seen_classes': 'top100_half',
        'pseudo_method': 'scene',
        'use_affordances': True
    })

    loaders,vocab = prepare_conceptgraph_dataloaders(dataloader_config, dataset_root=data_dir, caption_root=data_dir+'/data_download/complete_dataset/finetune_data/conceptgraph', cfg = dataset_cfg)

    cub_train_loader, cub_valid_loader, cub_test_loader = loaders['train'], loaders['val'], loaders['test']

    # CLIP_Net = load_model(device='cuda', model_path=None)
    device = 'cuda'
    CLIP_Net, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", "laion2b_s32b_b79k"
    )
    CLIP_Net = CLIP_Net.to(device)
    CLIP_Net.eval()
    clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

    adapter_cfg = mch({
        'ratio': 0.3,
        'mean_score': 0.2,
        'logit_scale': np.log(50),
        'embedding_dim': 1024,
    })

    train_clip_adapter(
        CLIP_Net,
        cub_train_loader,
        cub_test_loader,
        device='cuda',
        lr=1e-4,
        dtype=torch.cuda.FloatTensor,
        num_epochs=500,
        eval_every=10,
        ckpt_path=f'../../ckpt/12_Continual_LLM/no_captions_slow/',
        cfg=adapter_cfg
    )

######### TEST SIMPLE APPROACH ##########

if False:
    dataset_cfg = mch({
        'pseudo_label': 'caption',
        'pseudo_thresh': 0.29,
        'seen_classes': 'top100_half',
        'pseudo_method': 'scene',
        'use_affordances': False
    })

    dataloader_config = mch({
    'batch_size': 64,
    'random_erasing_prob': 0.,
    'traindata_shuffle': True,
    'loaders': ['test', 'val']
    })

    loaders,vocab = prepare_conceptgraph_dataloaders(dataloader_config, dataset_root=data_dir, caption_root=data_dir+'/data_download/complete_dataset/finetune_data/conceptgraph', cfg=dataset_cfg)

    cub_train_loader, cub_calib_loader, cub_test_loader = loaders['train'], loaders['val'], loaders['test']

    checkpoint_path = '../../ckpt/09_Conceptgraph_Init/thresh0.27_ratio0.3/best.pth'

    # CLIP_Net = load_model(device='cuda', model_path=None)
    device = 'cuda'
    CLIP_Net, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", "laion2b_s32b_b79k"
    )
    CLIP_Net = CLIP_Net.to(device)
    CLIP_Net.eval()
    clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

    adapter_cfg = mch({
        'ratio': 0.3,
        'mean_score': 0.2,
        'logit_scale': np.log(50),
        'embedding_dim': 1024,
    })

    test_clip_adapter(
        CLIP_Net,
        adapter_cfg,
        eval_loader=cub_test_loader,
        train_loader=cub_calib_loader,
        device='cuda',
        classifier='top_5',
        checkpoint_path=None
    )