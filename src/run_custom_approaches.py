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

######### TEST PROBVLM #########

# checkpoint_path = '../ckpt/ProbVLM_test/ProbVLM_Net_best.pth'

# loaders,vocab = prepare_scannet_dataloaders(dataloader_config, dataset_root=data_dir, caption_root=data_dir+'/data_download/complete_dataset/finetune_data/dataset_small')

# cub_train_loader, cub_valid_loader, cub_test_loader = loaders['train'], loaders['val'], loaders['test']

# CLIP_Net = load_model(device='cuda', model_path=None)
# ProbVLM_Net = BayesCap_for_CLIP(
#     inp_dim=512,
#     out_dim=512,
#     hid_dim=256,
#     num_layers=3,
#     p_drop=0.05,
# )

# test_ProbVLM(
#     CLIP_Net,
#     ProbVLM_Net,
#     cub_test_loader,
#     device='cuda',
#     dtype=torch.cuda.FloatTensor,
#     checkpoint_path=checkpoint_path
# )

######### TRAIN THE CLIP ADAPTER ##########

# TODO - add the proportion of classes used for training, and sweep it with and without pseudo-labels
# TODO - sweep pseudo_thresh
# TODO - review pseudo-labeling process (eg. pseudo-labelling happens in response to a query)

if False:
    dataset_cfg = mch({
        'pseudo_label': 'caption',
        'pseudo_thresh': 0.00,
        'seen_classes': 'top100_half',
        'pseudo_method': 'scene',
        'use_affordances': True
    })

    loaders,vocab = prepare_scannet_scene_dataloaders(dataloader_config, dataset_root=data_dir, caption_root=data_dir+'/data_download/complete_dataset/finetune_data/dataset01', cfg = dataset_cfg)

    cub_train_loader, cub_valid_loader, cub_test_loader = loaders['train'], loaders['val'], loaders['test']

    CLIP_Net = load_model(device='cuda', model_path=None)

    adapter_cfg = mch({
        'ratio': 0.3,
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
        ckpt_path=f'../ckpt/caption_tests/{strftime("%Y-%m-%d-%H:%M:%S", gmtime())}/',
        cfg=adapter_cfg
    )

######### TEST SIMPLE APPROACH ##########

if True:
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

    loaders,vocab = prepare_scannet_scene_dataloaders(dataloader_config, dataset_root=data_dir, caption_root=data_dir+'/data_download/complete_dataset/finetune_data/dataset01', cfg=dataset_cfg)

    cub_train_loader, cub_calib_loader, cub_test_loader = loaders['train'], loaders['val'], loaders['test']

    checkpoint_path = '../ckpt/08_Caption_Grid_Search/ratio0.3_thresh0.24/best.pth'

    CLIP_Net = load_model(device='cuda', model_path=None)

    adapter_cfg = mch({
        'ratio': 0.3,
        'mean_score': 0.2,
        'logit_scale': torch.log(torch.tensor(50.)),
        'embedding_dim': 512,
    })

    test_clip_adapter(
        CLIP_Net,
        adapter_cfg,
        eval_loader=cub_test_loader,
        train_loader=cub_calib_loader,
        device='cuda',
        classifier='top_5',
        checkpoint_path=checkpoint_path
    )