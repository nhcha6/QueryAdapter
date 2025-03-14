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
from clip_coop import *
import matplotlib.pyplot as plt
from time import gmtime, strftime
import open_clip
import clip

dataset = 'scannetpp' # coco or flickr

# !! Input dataset location
data_dir = ospj('/home/nicolas/hpc-home/Datasets/', dataset) # e.g. ospj(expanduser('~'), 'Documents', 'jm', 'data', dataset)
n_core_concepts = 0
lr = 0.5e-3
batch_size = 256

######### TRAIN THE CLIP ADAPTER ##########

# TODO - add the proportion of classes used for training, and sweep it with and without pseudo-labels
# TODO - sweep pseudo_thresh
# TODO - review pseudo-labeling process (eg. pseudo-labelling happens in response to a query)

dataset_cfg = mch({
    'pseudo_label': 'img_topk_ueo',
    'pseudo_thresh': 0.32,
    'seen_classes': 'top100_half',
    'pseudo_method': 'scene',
    'use_affordances': False,
    'n_core_concepts': n_core_concepts,
    'dataset_type': 'segments_medium',
    'n_topk': 8,
    'n_negatives': 100
})

dataloader_config = mch({
'batch_size': batch_size,
'random_erasing_prob': 0.,
'traindata_shuffle': True,
'loaders': ['train']
})

# CLIP_Net = load_model(device='cuda', model_path=None)
# device = 'cuda'
# CLIP_Net, _, clip_preprocess = open_clip.create_model_and_transforms(
#     "ViT-H-14", "laion2b_s32b_b79k"
# )
# CLIP_Net = CLIP_Net.to(device)
# CLIP_Net.eval()
# clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
# print(CLIP_Net.text_pool_type)

loaders,vocab = prepare_conceptgraph_dataloaders(dataloader_config, dataset_root=data_dir, caption_root=data_dir+'/data_download/complete_dataset/finetune_data/conceptgraph', cfg = dataset_cfg)

cub_train_loader, cub_valid_loader, cub_test_loader = loaders['train'], loaders['val'], loaders['test']

adapter_cfg = mch({
    'classnames': cub_train_loader.dataset.train_classes,
    'embedding_dim': 1024,
    'logit_scale': np.log(50),
})

# CLIP_Net = load_model(device='cuda', model_path=None)
device = 'cuda'
CLIP_Net, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-H-14", "laion2b_s32b_b79k"
)
CLIP_Net = CLIP_Net.to(device)
CLIP_Net.eval()
clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

# device = 'cuda'
# CLIP_Net, _ = clip.load('ViT-L/14')
# CLIP_Net = CLIP_Net.to(device)
# CLIP_Net.eval()

train_coop_ueo(
    CLIP_Net,
    cub_train_loader,
    cub_train_loader,
    device='cuda',
    lr=lr,
    dtype=torch.float32,
    num_epochs=110,
    eval_every=25,
    ckpt_path=f'../../output/{dataset_cfg["pseudo_label"]}_{dataset_cfg["dataset_type"]}_{batch_size}_{n_core_concepts}/',# always end with the n_core_concpets
    cfg=adapter_cfg,
    tokeniser=clip_tokenizer,
)

