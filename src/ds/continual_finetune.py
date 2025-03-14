"""CUB Caption image-to-caption retrieval dataset code

PCME
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import scipy.io
import glob
import cv2
from matplotlib import pyplot as plt
import tqdm
import clip
import torch
import torch.nn as nn
import h5py
import json
import open_clip
import random
from PIL import Image
import pickle
import torch.nn.functional as F
import sys
prefix = os.getcwd().split("ProbVLM")[0]
sys.path.append(f'{prefix}ProbVLM/src/tasks')
# sys.path.append('/home/n11223243/ProbVLM/src/tasks')
# from task_definition import *

def pad_text(num):
    if num<10:
        return '0000'+str(num)
    if num<100:
        return '000'+str(num)
           
    if num<1000:
        return '00'+str(num)

    
class ContinualPseudoLabel(Dataset):
    """CUB Captions Dataset.

    Args:
        image_root (string): Root directory where images are downloaded to.
        caption_root (string): Root directory where captions are downloaded to.
        target_classes (str or list): target class ids
            - if str, it is the name of the file with target classes (line by line)
            - if list, it is directly used to get classes
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        omit_ids (str, optional): Path of file with the list of image ids to omit,
            if not specified, use all images in the target classes.
        ids (str, optional): Path of file with the list of target image ids,
            if not specified, use all images in the target classes.
    """
    def __init__(self, image_root, caption_root,
                 split='val', pseudo_thresh=0.3, pseudo_method='scene', use_affordances=False, n_core_concepts=1
                 ):

        print('continual method')
    
        # Initialize the CLIP model used in conceptgraphs
        device = 'cuda'
        CLIP_Net, _, clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", "laion2b_s32b_b79k"
        )
        CLIP_Net = CLIP_Net.to(device)
        CLIP_Net.eval()
        clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

        # get the dataset splits
        if split in ['train', 'val']:
            dataset_split = split
        else:
            dataset_split = 'train'
        
        splits_path = image_root+f'/data_download/complete_dataset/splits/nvs_sem_{dataset_split}.txt'
        # read each line and add to the splits file
        with open(splits_path) as fin:
            scenes = [line.strip() for line in fin]
        # sort the scnes for consistency
        scenes.sort()
        # print(scenes)

        # get the semantic classes
        semantic_classes = image_root+f'/data_download/complete_dataset/metadata/semantic_classes.txt'
        top100_classes = image_root+f'/data_download/complete_dataset/metadata/semantic_benchmark/top100.txt'
        with open(semantic_classes) as fin:
            semantic_classes = [line.strip() for line in fin]
        # # print(semantic_classes)

        with open(top100_classes) as fin:
            top100_classes = [line.strip() for line in fin]
        # # print(top100_classes)

        # remove some classes
        exclude_classes = ['ceiling', 'floor', 'wall']
        target_classes = [cls for cls in semantic_classes if cls not in exclude_classes]
        top50  = [c for c in top100_classes[0:53] if c not in exclude_classes]

        # open the unambiguous affordances as a json file
        with open(image_root+f'/data_download/complete_dataset/metadata/top100_affordances.txt') as fin:
            top100_affordances = [line.strip() for line in fin]

        # open the descriptions as a json file
        with open(image_root+f'/data_download/complete_dataset/metadata/top100_descriptions.txt') as fin:
            top100_descriptions = [line.strip() for line in fin]

        # get the idx of exclude classes
        exclude_idx = [top100_classes.index(cls) for cls in exclude_classes]
        # remove the classes
        top100 = [top100_classes[idx] for idx in range(len(top100_classes)) if idx not in exclude_idx]
        top100_affordances = [top100_affordances[idx] for idx in range(len(top100_affordances)) if idx not in exclude_idx]
        top100_descriptions = [top100_descriptions[idx] for idx in range(len(top100_descriptions)) if idx not in exclude_idx]

        # # generate a list of core concepts or train_classes
        # top100_idx = [23, 37, 25, 41, 22, 7, 48, 46, 9, 3]
        # longtail_idx = [1233, 1586, 1180, 1047, 1020, 585, 231, 343, 101, 843, 1332, 1050, 1022, 1331, 1188, 228, 1181, 393, 1416, 201, 1178, 525, 709, 745, 798, 496, 1356, 1188, 368, 367, 1313, 885, 1284, 1134, 1597, 496, 1278, 153, 980, 1304, 1346, 850, 969, 978, 1542, 879, 1050, 102, 134, 1273, 736, 629, 1592, 1461, 625, 1462, 1413, 1063, 1105, 978, 838, 853, 1268, 248, 1262, 373, 885, 1368, 757, 747, 1284, 601, 570, 827, 1048, 147, 296, 1512, 873, 1011, 1264, 1224, 1048, 945, 1, 654, 393, 1557, 216, 495, 1110, 3, 562, 721, 1445, 1545, 350, 120, 27, 1447]
        # affordance_idx = [12, 3, 32, 37, 8, 11, 14, 18, 2, 31]
        # ignore_top_50 = [cls for cls in target_classes if cls not in top50]

        # # define the training classes
        # train_classes = [top100_classes[idx] for idx in top100_idx]
        # # add the longtail classes
        # train_classes += [ignore_top_50[idx] for idx in longtail_idx]
        # # add values from the affordances
        # train_classes += [affordance_list[idx] for idx in affordance_idx]
        # print(train_classes)

        # use all the classes
        train_classes = top100 + top100_affordances + top100_descriptions
        # select every 10th class for training
        train_classes = train_classes[::n_core_concepts]
        print(train_classes)
        print(len(train_classes))

        # encode the train classes
        with torch.no_grad():
            caption_query = [f'an image of a {concept}' for concept in train_classes]
            caption_targets = clip_tokenizer(caption_query)
            caption_targets = caption_targets.to(device)
            core_feats = CLIP_Net.encode_text(caption_targets)
                            
        # This can be editted dynamically as we change the training approach etc..
        scene_objects = {}
        scene_queries = {}
        pseudo_objects = {}
        # list all files in the caption root
        instance_files = glob.glob(caption_root+f'/{dataset_split}/*.h5')
        # iterate through each file
        for instance_file in tqdm.tqdm(instance_files):
            
            # open the object info file
            with h5py.File(instance_file, 'r') as f:
                try:
                    # get the object feature data
                    xfI = f['object_feats'][()]
                    caption = f['captions'][()].decode('utf-8')
                    scene_name = f['scene_name'][()].decode('utf-8')
                    core_classes = f['core_classes'][()]
                except Exception as e:
                    # print(e)
                    continue

            # if scene_name not in ['56a0ec536c', '8b5caf3398', '41b00feddb', '98b4ec142f', '7b6477cb95']:
            #     continue

            if scene_name not in scene_objects.keys():
                scene_objects[scene_name] = {
                    'object_feats': [],
                    'class_nums': [],
                }
                scene_queries[scene_name] = {
                    'captions': [],
                    'query_feats': [],
                    'object_ids': [],
                    'unseen': []
                }

            # caption with core concepts
            captions = []
            flag = [False]
            for core in core_classes:
                core = core.decode('utf-8')
                # check it is in the training classes
                if core in train_classes:
                    captions.append(f'an image of a {core}')
                    flag.append(True)

            # if empty, add the original object captions
            if not captions:
                # check for invalid captions
                if caption == 'invalid' or caption == '':
                    continue
                # otherwise label with caption
                captions = [f'an image of a {caption}']
                flag = [False]

            # print(captions)
            
            # append the object
            visual_features = np.expand_dims(xfI, axis=0)

            # append the object
            scene_objects[scene_name]['object_feats'].append(visual_features)
            object_idx = len(scene_objects[scene_name]['object_feats']) - 1

            # get the caption features
            with torch.no_grad():
                caption_targets = clip_tokenizer(captions)
                caption_targets = caption_targets.to(device)
                caption_feats = CLIP_Net.encode_text(caption_targets)

            # iterate through each caption
            for i in range(len(captions)): 
                caption = captions[i]  
                xfT = caption_feats[i].unsqueeze(0).cpu().numpy() 
                unseen = flag[i]                  

                # check if caption not in the data structure
                if caption not in scene_queries[scene_name]['captions']:
                    scene_queries[scene_name]['captions'].append(caption)
                    scene_queries[scene_name]['query_feats'].append(xfT)
                    scene_queries[scene_name]['object_ids'].append([str(object_idx)])
                    scene_queries[scene_name]['unseen'].append(unseen)
                else:
                    scene_queries[scene_name]['object_ids'][scene_queries[scene_name]['captions'].index(caption)].append(str(object_idx))

        # query_feats: text feature of unique class in the scene
        # captions: captions of unique class in the scene for explainability
        # object_feats: visual feature of all unique object instances
        # query_labels: 0 or 1 for each object instance

        # get the maximum number of objects in a single scene, to pad the object features
        self.max_objects = max([len(scene_objects[scene]['object_feats']) for scene in scene_objects.keys()])

        self.query_feats = []
        self.captions = []
        self.object_feats = []
        self.query_labels = []
        self.unseen = []
        self.n_objects = 0
        self.n_queries = 0
        for scene in scene_queries.keys():
            # check there is at least one object
            if not scene_objects[scene]['object_feats']:
                continue
            self.n_objects+=len(scene_objects[scene]['object_feats'])
            
            object_feats = scene_objects[scene]['object_feats']

            for i in range(len(scene_queries[scene]['query_feats'])):
                # append the query features
                self.query_feats.append(scene_queries[scene]['query_feats'][i])
                self.captions.append(scene_queries[scene]['captions'][i])
                # self.query_labels.append(scene_queries[scene]['object_ids'][i])
                self.query_labels.append('_'.join(scene_queries[scene]['object_ids'][i]))
                # append the object features
                self.object_feats.append(object_feats)
                # append the top100
                self.unseen.append(scene_queries[scene]['unseen'][i])
                self.n_queries+=1

        # for k in range(len(self.query_labels)):
        #     print(self.query_labels[k])

    def __getitem__(self, index):
        # print(self.captions[index])
        # print(self.query_feats[index].shape)
        # print(self.query_labels[index])
        # print(self.object_feats[index].shape)
        
        # need to convert the object_feats to a tensor
        object_feats = self.object_feats[index]
        # iterate through each element and calculate the mean tensor
        # object_feats = [np.mean(feats, axis=0) for feats in object_feats]
        # alternatively, we can iterate over the object feats and randomly select one features. This is useful for training image models
        object_feats = [feats[np.random.randint(feats.shape[0])] for feats in object_feats]

        # stack the object_feats
        object_feats = np.stack(object_feats)
        # pad with zeros to make object feats (max_objects, 512)
        object_feats = np.pad(object_feats, ((0, self.max_objects-object_feats.shape[0]), (0,0)), 'constant', constant_values=0)
        
        # print(self.query_feats[index].shape)
        # print(self.query_labels[index])
        # print(self.captions[index])
        # print(object_feats.shape)
        # print(self.unseen[index])

        return self.query_feats[index], self.query_labels[index], self.captions[index], object_feats, self.unseen[index]
    
    def __len__(self):
        return len(self.captions)
    
class SegmentPseudoLabel(Dataset):
    """CUB Captions Dataset.

    Args:
        image_root (string): Root directory where images are downloaded to.
        caption_root (string): Root directory where captions are downloaded to.
        target_classes (str or list): target class ids
            - if str, it is the name of the file with target classes (line by line)
            - if list, it is directly used to get classes
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        omit_ids (str, optional): Path of file with the list of image ids to omit,
            if not specified, use all images in the target classes.
        ids (str, optional): Path of file with the list of target image ids,
            if not specified, use all images in the target classes.
    """
    def __init__(self, image_root, caption_root,
                 split='val', pseudo_thresh=0.3, pseudo_method='scene', use_affordances=False, n_core_concepts=1
                 ):

        print(f'segments method')
    
        # Initialize the CLIP model used in conceptgraphs
        device = 'cuda'
        CLIP_Net, _, clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", "laion2b_s32b_b79k"
        )
        CLIP_Net = CLIP_Net.to(device)
        CLIP_Net.eval()
        clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

        # get the dataset splits
        if split in ['train', 'val']:
            dataset_split = split
        else:
            dataset_split = 'train'
        
        splits_path = image_root+f'/data_download/complete_dataset/splits/nvs_sem_{dataset_split}.txt'
        # read each line and add to the splits file
        with open(splits_path) as fin:
            scenes = [line.strip() for line in fin]
        # sort the scnes for consistency
        scenes.sort()
        # print(scenes)

        # get the semantic classes
        semantic_classes = image_root+f'/data_download/complete_dataset/metadata/semantic_classes.txt'
        top100_classes = image_root+f'/data_download/complete_dataset/metadata/semantic_benchmark/top100.txt'
        with open(semantic_classes) as fin:
            semantic_classes = [line.strip() for line in fin]
        # # print(semantic_classes)

        with open(top100_classes) as fin:
            top100_classes = [line.strip() for line in fin]
        # # print(top100_classes)

        # remove some classes
        exclude_classes = ['ceiling', 'floor', 'wall']
        target_classes = [cls for cls in semantic_classes if cls not in exclude_classes]
        top50  = [c for c in top100_classes[0:53] if c not in exclude_classes]

        # open the unambiguous affordances as a json file
        with open(image_root+f'/data_download/complete_dataset/metadata/top100_affordances.txt') as fin:
            top100_affordances = [line.strip() for line in fin]

        # open the descriptions as a json file
        with open(image_root+f'/data_download/complete_dataset/metadata/top100_descriptions.txt') as fin:
            top100_descriptions = [line.strip() for line in fin]

        # get the idx of exclude classes
        exclude_idx = [top100_classes.index(cls) for cls in exclude_classes]
        # remove the classes
        top100 = [top100_classes[idx] for idx in range(len(top100_classes)) if idx not in exclude_idx]
        top100_affordances = [top100_affordances[idx] for idx in range(len(top100_affordances)) if idx not in exclude_idx]
        top100_descriptions = [top100_descriptions[idx] for idx in range(len(top100_descriptions)) if idx not in exclude_idx]

        # just select the top 50 classes
        top100 = top100[0:50]
        top100_affordances = top100_affordances[0:50]
        top100_descriptions = top100_descriptions[0:50]

        # # generate a list of core concepts or train_classes
        # top100_idx = [23, 37, 25, 41, 22, 7, 48, 46, 9, 3]
        # longtail_idx = [1233, 1586, 1180, 1047, 1020, 585, 231, 343, 101, 843, 1332, 1050, 1022, 1331, 1188, 228, 1181, 393, 1416, 201, 1178, 525, 709, 745, 798, 496, 1356, 1188, 368, 367, 1313, 885, 1284, 1134, 1597, 496, 1278, 153, 980, 1304, 1346, 850, 969, 978, 1542, 879, 1050, 102, 134, 1273, 736, 629, 1592, 1461, 625, 1462, 1413, 1063, 1105, 978, 838, 853, 1268, 248, 1262, 373, 885, 1368, 757, 747, 1284, 601, 570, 827, 1048, 147, 296, 1512, 873, 1011, 1264, 1224, 1048, 945, 1, 654, 393, 1557, 216, 495, 1110, 3, 562, 721, 1445, 1545, 350, 120, 27, 1447]
        # affordance_idx = [12, 3, 32, 37, 8, 11, 14, 18, 2, 31]
        # ignore_top_50 = [cls for cls in target_classes if cls not in top50]

        # # define the training classes
        # train_classes = [top100_classes[idx] for idx in top100_idx]
        # # add the longtail classes
        # train_classes += [ignore_top_50[idx] for idx in longtail_idx]
        # # add values from the affordances
        # train_classes += [affordance_list[idx] for idx in affordance_idx]
        # print(train_classes)

        # use all the classes
        # train_classes = top100[1::n_core_concepts] + top100_affordances[0::n_core_concepts] + top100_descriptions[1::n_core_concepts]
        # select every 10th class for training
        # train_classes = train_classes[1::n_core_concepts]

        # use all the classes
        if use_affordances:
            # train_classes = top100_affordances[n_core_concepts::10] #+ top100_affordances[0::n_core_concepts] + top100_descriptions[1::n_core_concepts]
            train_classes = top100_affordances[0:48][n_core_concepts::8] #+ top100_affordances[0::n_core_concepts] + top100_descriptions[1::n_core_concepts]
        else:
            # train_classes = top100[n_core_concepts::10] #+ top100_affordances[0::n_core_concepts] + top100_descriptions[1::n_core_concepts]
            train_classes = top100[0:48][n_core_concepts::8] #+ top100_affordances[0::n_core_concepts] + top100_descriptions[1::n_core_concepts]

        print(train_classes)
        print(len(train_classes))
        self.train_classes = train_classes

        # encode the train classes
        with torch.no_grad():
            caption_query = [f'An image of a {concept}' for concept in train_classes]
            caption_targets = clip_tokenizer(caption_query)
            caption_targets = caption_targets.to(device)
            core_feats = CLIP_Net.encode_text(caption_targets)
                            
        # This can be editted dynamically as we change the training approach etc..
        scene_objects = {}
        scene_queries = {}
        pseudo_objects = {}
        # list all files in the caption root
        instance_files = glob.glob(caption_root+f'/{dataset_split}/*.pkl')
        # iterate through each file
        for instance_file in tqdm.tqdm(instance_files):
            
            # # open the object info file
            # with h5py.File(instance_file, 'r') as f:
            #     try:
            #         # get the object feature data
            #         xfI = f['object_feats'][()]
            #         caption = f['captions'][()].decode('utf-8')
            #         scene_name = f['scene_name'][()].decode('utf-8')
            #         core_classes = f['core_classes'][()]
            #     except Exception as e:
            #         # print(e)
            #         continue

            scene_name = instance_file.split('/')[-1].split('.')[0]

            # if scene_name not in ['56a0ec536c', '8b5caf3398', '41b00feddb', '98b4ec142f', '7b6477cb95']:
            #     continue

            # open the pkl file
            with open(instance_file, 'rb') as f:
                segmented_objects = pickle.load(f)

            for obj_idx in range(len(segmented_objects['feature'])):
                xfI = segmented_objects['feature'][obj_idx]
                caption = segmented_objects['caption'][obj_idx]
                core_classes = segmented_objects['core_classes'][obj_idx]
                image_path = segmented_objects['image_path'][obj_idx]
                bbox = segmented_objects['bbox'][obj_idx][0]

                # check if this is a new scene
                if scene_name not in scene_objects.keys():
                    scene_objects[scene_name] = {
                        'object_feats': [],
                        'class_nums': [],
                        'captions': [],
                        'query_feats': [],
                        'unseen': []
                    }
                    scene_queries[scene_name] = {
                        'captions': [],
                        'query_feats': [],
                        'object_ids': [],
                        'unseen': []
                    }

                # if not core_classes:
                #     continue

                # check for invalid captions
                if caption == 'invalid' or caption == '':
                    continue

                if pseudo_method == 'segments':
                    captions = [caption]
                    flag = [True]
                else:
                    captions = []
                    flag = []   

                # caption with core concepts
                for core in core_classes:
                    # check it is in the training classes
                    if core in train_classes:
                        captions.append(f'An image of a {core}')
                        flag.append(False)

                        # print(core)
                        # # open the image
                        # image = Image.open(image_path)
                        # image = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                        # # display with matplotlib
                        # plt.figure()
                        # plt.imshow(image)
                        # plt.show()

                # # if empty, add the original object captions
                # if not captions:
                #     # otherwise label with caption
                #     captions = [caption]
                #     flag = [True]

                # continue if no captions
                if not captions:
                    continue

                # append the object
                visual_features = np.expand_dims(xfI, axis=0)

                # append the object
                scene_objects[scene_name]['object_feats'].append(visual_features)
                scene_objects[scene_name]['captions'].append([])
                scene_objects[scene_name]['query_feats'].append([])
                scene_objects[scene_name]['unseen'].append([])
                object_idx = len(scene_objects[scene_name]['object_feats']) - 1
    
                # get the caption features
                with torch.no_grad():
                    caption_targets = clip_tokenizer(captions)
                    caption_targets = caption_targets.to(device)
                    caption_feats = CLIP_Net.encode_text(caption_targets)

                # iterate through each caption
                for i in range(len(captions)): 
                    caption = captions[i]  
                    xfT = caption_feats[i].unsqueeze(0).cpu().numpy() 
                    unseen = flag[i]

                    # append to the object
                    scene_objects[scene_name]['captions'][object_idx].append(caption)
                    scene_objects[scene_name]['query_feats'][object_idx].append(xfT)     
                    scene_objects[scene_name]['unseen'][object_idx].append(unseen)             

                    # check if caption not in the data structure
                    if caption not in scene_queries[scene_name]['captions']:
                        scene_queries[scene_name]['captions'].append(caption)
                        scene_queries[scene_name]['query_feats'].append(xfT)
                        scene_queries[scene_name]['object_ids'].append([str(object_idx)])
                        scene_queries[scene_name]['unseen'].append(unseen)
                    else:
                        scene_queries[scene_name]['object_ids'][scene_queries[scene_name]['captions'].index(caption)].append(str(object_idx))

        # query_feats: text feature of unique class in the scene
        # captions: captions of unique class in the scene for explainability
        # object_feats: visual feature of all unique object instances
        # query_labels: 0 or 1 for each object instance

        # create object data accessible in the get_item
        self.objects = {
            'object_feats': [],
            'captions': [],
            'query_feats': [],
            'unseen': []
        }
        for scene in scene_objects.keys():
            # for each object in the scene
            for obj_idx in range(len(scene_objects[scene]['object_feats'])):
                self.objects['object_feats'].append(scene_objects[scene]['object_feats'][obj_idx])
                self.objects['captions'].append(scene_objects[scene]['captions'][obj_idx])
                self.objects['query_feats'].append(scene_objects[scene]['query_feats'][obj_idx])
                self.objects['unseen'].append(scene_objects[scene]['unseen'][obj_idx])

        # get the maximum number of objects in a single scene, to pad the object features
        self.max_objects = max([len(scene_objects[scene]['object_feats']) for scene in scene_objects.keys()])

        self.query_feats = []
        self.captions = []
        self.object_feats = []
        self.query_labels = []
        self.unseen = []
        self.n_objects = 0
        self.n_queries = 0
        for scene in scene_queries.keys():
            # check there is at least one object
            if not scene_objects[scene]['object_feats']:
                continue
            self.n_objects+=len(scene_objects[scene]['object_feats'])
            
            object_feats = scene_objects[scene]['object_feats']

            for i in range(len(scene_queries[scene]['query_feats'])):
                # append the query features
                self.query_feats.append(scene_queries[scene]['query_feats'][i])
                self.captions.append(scene_queries[scene]['captions'][i])
                # self.query_labels.append(scene_queries[scene]['object_ids'][i])
                self.query_labels.append('_'.join(scene_queries[scene]['object_ids'][i]))
                # append the object features
                self.object_feats.append(object_feats)
                # append the top100
                self.unseen.append(scene_queries[scene]['unseen'][i])
                self.n_queries+=1

        # for k in range(len(self.query_labels)):
        #     print(self.query_labels[k])

    def __getitem__(self, index):
        # get the object features
        object_feat = self.objects['object_feats'][index]
        # randomly select one of the captions and query features
        caption_idx = np.random.randint(len(self.objects['captions'][index]))
        caption = self.objects['captions'][index][caption_idx]
        query_feat = self.objects['query_feats'][index][caption_idx]
        unseen = self.objects['unseen'][index][caption_idx]

        object_feats = [object_feat]
        query_labels = '0'
        while len(object_feats) < 50:
            # randomly select an object
            random_int = np.random.randint(len(self.objects['object_feats']))
            # check caption is not in the object
            if caption in self.objects['captions'][random_int]:
                continue
            # add the feature to the object_feats
            object_feats.append(self.objects['object_feats'][random_int])
                
        # object_feats = self.object_feats[index]
        # query_labels = self.query_labels[index].split('_')
        # query_caption = self.captions[index]
        # # get just the feats that are in the query labels
        # object_feats = [object_feats[int(idx)] for idx in query_labels]
        # # query feats
        # query_labels = '_'.join([str(idx) for idx in range(len(query_labels))])

        # # add other objects until we have 50
        # while len(object_feats) < 50:
        #     random_int = random.randint(0, len(self.object_feats)-1)
        #     random_caption = self.captions[random_int]
        #     # make sure we are getting an object with a different label
        #     if random_caption == query_caption:
        #         continue
        #     # select the first object from the query labels
        #     random_object = self.query_labels[random_int].split('_')[0]
        #     # get the object features
        #     random_feats = self.object_feats[random_int][int(random_object)]
        #     object_feats.append(random_feats)

        # stack the object_feats
        object_feats = np.stack(object_feats).squeeze(1)

        # return self.query_feats[index], query_labels, self.captions[index], object_feats, self.unseen[index]
        return query_feat, query_labels, caption, object_feats, unseen

    # def __getitem__(self, index):
        
    #     object_feats = self.object_feats[index]
    #     query_labels = self.query_labels[index].split('_')
    #     query_caption = self.captions[index]
    #     # get just the feats that are in the query labels
    #     object_feats = [object_feats[int(idx)] for idx in query_labels]
    #     # query feats
    #     query_labels = '_'.join([str(idx) for idx in range(len(query_labels))])

    #     # add other objects until we have 50
    #     while len(object_feats) < 50:
    #         random_int = random.randint(0, len(self.object_feats)-1)
    #         random_caption = self.captions[random_int]
    #         # make sure we are getting an object with a different label
    #         if random_caption == query_caption:
    #             continue
    #         # select the first object from the query labels
    #         random_object = self.query_labels[random_int].split('_')[0]
    #         # get the object features
    #         random_feats = self.object_feats[random_int][int(random_object)]
    #         object_feats.append(random_feats)

    #     # stack the object_feats
    #     object_feats = np.stack(object_feats).squeeze(1)

    #     return self.query_feats[index], query_labels, self.captions[index], object_feats, self.unseen[index]
        
    # def __getitem__(self, index):
    #     # print(self.captions[index])
    #     # print(self.query_feats[index].shape)
    #     # print(self.query_labels[index])
    #     # print(self.object_feats[index].shape)
        
    #     # need to convert the object_feats to a tensor
    #     object_feats = self.object_feats[index]
    #     # iterate through each element and calculate the mean tensor
    #     # object_feats = [np.mean(feats, axis=0) for feats in object_feats]
    #     # alternatively, we can iterate over the object feats and randomly select one features. This is useful for training image models
    #     object_feats = [feats[np.random.randint(feats.shape[0])] for feats in object_feats]

    #     # get 50 random indices for the object_feats
    #     indices = self.query_labels[index].split('_')
    #     indices = [int(idx) for idx in indices]
    #     while len(indices) < 50:
    #         random_int = random.randint(0, len(object_feats)-1)
    #         if random_int not in indices:
    #             indices.append(random_int)
    #     object_feats = [object_feats[idx] for idx in indices]
    #     new_query_label = '_'.join([str(idx) for idx in range(len(self.query_labels[index].split('_')))])
    #     self.query_labels[index] = new_query_label

    #     # stack the object_feats
    #     object_feats = np.stack(object_feats)
    #     # pad with zeros to make object feats (max_objects, 512)
    #     # object_feats = np.pad(object_feats, ((0, self.max_objects-object_feats.shape[0]), (0,0)), 'constant', constant_values=0)
        
    #     # print(self.query_feats[index].shape)
    #     # print(self.query_labels[index])
    #     # print(self.captions[index])
    #     # print(object_feats.shape)
    #     # print(self.unseen[index])

    #     return self.query_feats[index], self.query_labels[index], self.captions[index], object_feats, self.unseen[index]
    
    def __len__(self):
        # return len(self.captions)
        return len(self.objects['captions'])

class SegmentAlternativeLabels(Dataset):
    """CUB Captions Dataset.

    Args:
        image_root (string): Root directory where images are downloaded to.
        caption_root (string): Root directory where captions are downloaded to.
        target_classes (str or list): target class ids
            - if str, it is the name of the file with target classes (line by line)
            - if list, it is directly used to get classes
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        omit_ids (str, optional): Path of file with the list of image ids to omit,
            if not specified, use all images in the target classes.
        ids (str, optional): Path of file with the list of target image ids,
            if not specified, use all images in the target classes.
    """
    def __init__(self, image_root, caption_root,
                 split='val', pseudo_thresh=0.3, pseudo_method='scene', use_affordances=False, n_core_concepts=1
                 ):

        print(f'{pseudo_method} method')
    
        # Initialize the CLIP model used in conceptgraphs
        device = 'cuda'
        CLIP_Net, _, clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", "laion2b_s32b_b79k"
        )
        CLIP_Net = CLIP_Net.to(device)
        CLIP_Net.eval()
        clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

        # get the dataset splits
        if split in ['train', 'val']:
            dataset_split = split
        else:
            dataset_split = 'train'
        
        splits_path = image_root+f'/data_download/complete_dataset/splits/nvs_sem_{dataset_split}.txt'
        # read each line and add to the splits file
        with open(splits_path) as fin:
            scenes = [line.strip() for line in fin]
        # sort the scnes for consistency
        scenes.sort()
        # print(scenes)

        # get the semantic classes
        semantic_classes = image_root+f'/data_download/complete_dataset/metadata/semantic_classes.txt'
        top100_classes = image_root+f'/data_download/complete_dataset/metadata/semantic_benchmark/top100.txt'
        with open(semantic_classes) as fin:
            semantic_classes = [line.strip() for line in fin]
        # # print(semantic_classes)

        with open(top100_classes) as fin:
            top100_classes = [line.strip() for line in fin]
        # # print(top100_classes)

        # remove some classes
        exclude_classes = ['ceiling', 'floor', 'wall']
        target_classes = [cls for cls in semantic_classes if cls not in exclude_classes]
        top50  = [c for c in top100_classes[0:53] if c not in exclude_classes]

        # open the unambiguous affordances as a json file
        with open(image_root+f'/data_download/complete_dataset/metadata/top100_affordances.txt') as fin:
            top100_affordances = [line.strip() for line in fin]

        # open the descriptions as a json file
        with open(image_root+f'/data_download/complete_dataset/metadata/top100_descriptions.txt') as fin:
            top100_descriptions = [line.strip() for line in fin]

        # get the idx of exclude classes
        exclude_idx = [top100_classes.index(cls) for cls in exclude_classes]
        # remove the classes
        top100 = [top100_classes[idx] for idx in range(len(top100_classes)) if idx not in exclude_idx]
        top100_affordances = [top100_affordances[idx] for idx in range(len(top100_affordances)) if idx not in exclude_idx]
        top100_descriptions = [top100_descriptions[idx] for idx in range(len(top100_descriptions)) if idx not in exclude_idx]

        # just select the top 50 classes
        top100 = top100[0:50]
        top100_affordances = top100_affordances[0:50]
        top100_descriptions = top100_descriptions[0:50]

        # # generate a list of core concepts or train_classes
        # top100_idx = [23, 37, 25, 41, 22, 7, 48, 46, 9, 3]
        # longtail_idx = [1233, 1586, 1180, 1047, 1020, 585, 231, 343, 101, 843, 1332, 1050, 1022, 1331, 1188, 228, 1181, 393, 1416, 201, 1178, 525, 709, 745, 798, 496, 1356, 1188, 368, 367, 1313, 885, 1284, 1134, 1597, 496, 1278, 153, 980, 1304, 1346, 850, 969, 978, 1542, 879, 1050, 102, 134, 1273, 736, 629, 1592, 1461, 625, 1462, 1413, 1063, 1105, 978, 838, 853, 1268, 248, 1262, 373, 885, 1368, 757, 747, 1284, 601, 570, 827, 1048, 147, 296, 1512, 873, 1011, 1264, 1224, 1048, 945, 1, 654, 393, 1557, 216, 495, 1110, 3, 562, 721, 1445, 1545, 350, 120, 27, 1447]
        # affordance_idx = [12, 3, 32, 37, 8, 11, 14, 18, 2, 31]
        # ignore_top_50 = [cls for cls in target_classes if cls not in top50]

        # # define the training classes
        # train_classes = [top100_classes[idx] for idx in top100_idx]
        # # add the longtail classes
        # train_classes += [ignore_top_50[idx] for idx in longtail_idx]
        # # add values from the affordances
        # train_classes += [affordance_list[idx] for idx in affordance_idx]
        # print(train_classes)

        # use all the classes
        train_classes = top100[1::n_core_concepts] + top100_affordances[0::n_core_concepts] + top100_descriptions[1::n_core_concepts]
        # select every 10th class for training
        # train_classes = train_classes[1::n_core_concepts]
        print(train_classes)
        print(len(train_classes))

        # encode the train classes
        with torch.no_grad():
            caption_query = [f'An image of a {concept}' for concept in train_classes]
            caption_targets = clip_tokenizer(caption_query)
            caption_targets = caption_targets.to(device)
            core_feats = CLIP_Net.encode_text(caption_targets)
                            
        # This can be editted dynamically as we change the training approach etc..
        scene_objects = {}
        scene_queries = {}
        pseudo_objects = {}
        # list all files in the caption root
        instance_files = glob.glob(caption_root+f'/{dataset_split}/*.pkl')
        # iterate through each file
        for instance_file in tqdm.tqdm(instance_files):
            
            # # open the object info file
            # with h5py.File(instance_file, 'r') as f:
            #     try:
            #         # get the object feature data
            #         xfI = f['object_feats'][()]
            #         caption = f['captions'][()].decode('utf-8')
            #         scene_name = f['scene_name'][()].decode('utf-8')
            #         core_classes = f['core_classes'][()]
            #     except Exception as e:
            #         # print(e)
            #         continue

            scene_name = instance_file.split('/')[-1].split('.')[0]

            # if scene_name not in ['56a0ec536c', '8b5caf3398', '41b00feddb', '98b4ec142f', '7b6477cb95']:
            #     continue

            # open the pkl file
            with open(instance_file, 'rb') as f:
                segmented_objects = pickle.load(f)

            for obj_idx in range(len(segmented_objects['feature'])):
                xfI = segmented_objects['feature'][obj_idx]
                caption = segmented_objects['caption'][obj_idx]
                core_classes = segmented_objects['core_classes'][obj_idx]
                image_path = segmented_objects['image_path'][obj_idx]
                bbox = segmented_objects['bbox'][obj_idx][0]

                # check if this is a new scene
                if scene_name not in scene_objects.keys():
                    scene_objects[scene_name] = {
                        'object_feats': [],
                        'class_nums': [],
                        'captions': [],
                        'query_feats': [],
                        'unseen': []
                    }
                    scene_queries[scene_name] = {
                        'captions': [],
                        'query_feats': [],
                        'object_ids': [],
                        'unseen': []
                    }

                # if not core_classes:
                #     continue

                # check for invalid captions
                if caption == 'invalid' or caption == '':
                    continue

                # caption with core concepts
                if pseudo_method == 'caption_only':
                    captions = [caption]
                    flag = [True]
                
                # if pseudo_method is cosine sim
                elif pseudo_method == 'cosine_sim':
                    # normalise the core feeatures
                    core_feats_norm = core_feats / core_feats.norm(dim=-1, keepdim=True)
                    # normalise the object features
                    object_feats_norm = xfI / xfI.norm(dim=-1, keepdim=True)
                    # put tensors on device
                    object_feats_norm = object_feats_norm.to(device)
                    # calculate the cosine similarity
                    sim = F.cosine_similarity(core_feats_norm, object_feats_norm, dim=-1)
                    # get all the concepts with a similarity above the threshold
                    above_thresh = sim > pseudo_thresh
                    # get all concepts above the threshold
                    core_concepts = [train_classes[idx] for idx in range(len(train_classes)) if above_thresh[idx]]

                    captions = []
                    flag = []
                    for concept in core_concepts:
                        captions.append(f'An image of a {concept}')
                        flag.append(False)

                # for core in core_classes:
                #     # check it is in the training classes
                #     if core in train_classes:
                #         captions.append(f'An image of a {core}')
                #         flag.append(False)

                # print(captions)
                # # open the image
                # image = Image.open(image_path)
                # image = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                # # display with matplotlib
                # plt.figure()
                # plt.imshow(image)
                # plt.show()

                # skip objects with no captions
                if not captions:
                    continue

                # # if empty, add the original object captions
                # if not captions:
                #     # otherwise label with caption
                #     captions = [caption]
                #     flag = [True]

                # append the object
                visual_features = np.expand_dims(xfI, axis=0)

                # append the object
                scene_objects[scene_name]['object_feats'].append(visual_features)
                scene_objects[scene_name]['captions'].append([])
                scene_objects[scene_name]['query_feats'].append([])
                scene_objects[scene_name]['unseen'].append([])
                object_idx = len(scene_objects[scene_name]['object_feats']) - 1
    
                # get the caption features
                with torch.no_grad():
                    caption_targets = clip_tokenizer(captions)
                    caption_targets = caption_targets.to(device)
                    caption_feats = CLIP_Net.encode_text(caption_targets)

                # iterate through each caption
                for i in range(len(captions)): 
                    caption = captions[i]  
                    xfT = caption_feats[i].unsqueeze(0).cpu().numpy() 
                    unseen = flag[i]

                    # append to the object
                    scene_objects[scene_name]['captions'][object_idx].append(caption)
                    scene_objects[scene_name]['query_feats'][object_idx].append(xfT)     
                    scene_objects[scene_name]['unseen'][object_idx].append(unseen)             

                    # check if caption not in the data structure
                    if caption not in scene_queries[scene_name]['captions']:
                        scene_queries[scene_name]['captions'].append(caption)
                        scene_queries[scene_name]['query_feats'].append(xfT)
                        scene_queries[scene_name]['object_ids'].append([str(object_idx)])
                        scene_queries[scene_name]['unseen'].append(unseen)
                    else:
                        scene_queries[scene_name]['object_ids'][scene_queries[scene_name]['captions'].index(caption)].append(str(object_idx))

        # query_feats: text feature of unique class in the scene
        # captions: captions of unique class in the scene for explainability
        # object_feats: visual feature of all unique object instances
        # query_labels: 0 or 1 for each object instance

        # create object data accessible in the get_item
        self.objects = {
            'object_feats': [],
            'captions': [],
            'query_feats': [],
            'unseen': []
        }
        for scene in scene_objects.keys():
            # for each object in the scene
            for obj_idx in range(len(scene_objects[scene]['object_feats'])):
                self.objects['object_feats'].append(scene_objects[scene]['object_feats'][obj_idx])
                self.objects['captions'].append(scene_objects[scene]['captions'][obj_idx])
                self.objects['query_feats'].append(scene_objects[scene]['query_feats'][obj_idx])
                self.objects['unseen'].append(scene_objects[scene]['unseen'][obj_idx])

        # get the maximum number of objects in a single scene, to pad the object features
        self.max_objects = max([len(scene_objects[scene]['object_feats']) for scene in scene_objects.keys()])

        self.query_feats = []
        self.captions = []
        self.object_feats = []
        self.query_labels = []
        self.unseen = []
        self.n_objects = 0
        self.n_queries = 0
        for scene in scene_queries.keys():
            # check there is at least one object
            if not scene_objects[scene]['object_feats']:
                continue
            self.n_objects+=len(scene_objects[scene]['object_feats'])
            
            object_feats = scene_objects[scene]['object_feats']

            for i in range(len(scene_queries[scene]['query_feats'])):
                # append the query features
                self.query_feats.append(scene_queries[scene]['query_feats'][i])
                self.captions.append(scene_queries[scene]['captions'][i])
                # self.query_labels.append(scene_queries[scene]['object_ids'][i])
                self.query_labels.append('_'.join(scene_queries[scene]['object_ids'][i]))
                # append the object features
                self.object_feats.append(object_feats)
                # append the top100
                self.unseen.append(scene_queries[scene]['unseen'][i])
                self.n_queries+=1

        # for k in range(len(self.query_labels)):
        #     print(self.query_labels[k])

    def __getitem__(self, index):
        # get the object features
        object_feat = self.objects['object_feats'][index]
        # randomly select one of the captions and query features
        caption_idx = np.random.randint(len(self.objects['captions'][index]))
        caption = self.objects['captions'][index][caption_idx]
        query_feat = self.objects['query_feats'][index][caption_idx]
        unseen = self.objects['unseen'][index][caption_idx]

        object_feats = [object_feat]
        query_labels = '0'
        while len(object_feats) < 50:
            # randomly select an object
            random_int = np.random.randint(len(self.objects['object_feats']))
            # check caption is not in the object
            if caption in self.objects['captions'][random_int]:
                continue
            # add the feature to the object_feats
            object_feats.append(self.objects['object_feats'][random_int])

        # stack the object_feats
        object_feats = np.stack(object_feats).squeeze(1)

        # print(query_feat.shape)
        # print(query_labels)
        # print(caption)
        # print(object_feats.shape)
        # print(unseen)

        # return self.query_feats[index], query_labels, self.captions[index], object_feats, self.unseen[index]
        return query_feat, query_labels, caption, object_feats, unseen

    def __len__(self):
        # return len(self.captions)
        return len(self.objects['captions'])

class SegmentTopkLabels(Dataset):
    """CUB Captions Dataset.

    Args:
        image_root (string): Root directory where images are downloaded to.
        caption_root (string): Root directory where captions are downloaded to.
        target_classes (str or list): target class ids
            - if str, it is the name of the file with target classes (line by line)
            - if list, it is directly used to get classes
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        omit_ids (str, optional): Path of file with the list of image ids to omit,
            if not specified, use all images in the target classes.
        ids (str, optional): Path of file with the list of target image ids,
            if not specified, use all images in the target classes.
    """
    def __init__(self, image_root, caption_root,
                 split='val', pseudo_thresh=0.3, pseudo_method='scene', use_affordances=False, n_core_concepts=1
                 ):

        print(f'top_k')
    
        # Initialize the CLIP model used in conceptgraphs
        device = 'cuda'
        CLIP_Net, _, clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", "laion2b_s32b_b79k"
        )
        CLIP_Net = CLIP_Net.to(device)
        CLIP_Net.eval()
        clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

        # get the dataset splits
        if split in ['train', 'val']:
            dataset_split = split
        else:
            dataset_split = 'train'
        
        splits_path = image_root+f'/data_download/complete_dataset/splits/nvs_sem_{dataset_split}.txt'
        # read each line and add to the splits file
        with open(splits_path) as fin:
            scenes = [line.strip() for line in fin]
        # sort the scnes for consistency
        scenes.sort()
        # print(scenes)

        # get the semantic classes
        semantic_classes = image_root+f'/data_download/complete_dataset/metadata/semantic_classes.txt'
        top100_classes = image_root+f'/data_download/complete_dataset/metadata/semantic_benchmark/top100.txt'
        with open(semantic_classes) as fin:
            semantic_classes = [line.strip() for line in fin]
        # # print(semantic_classes)

        with open(top100_classes) as fin:
            top100_classes = [line.strip() for line in fin]
        # # print(top100_classes)

        # remove some classes
        exclude_classes = ['ceiling', 'floor', 'wall']
        target_classes = [cls for cls in semantic_classes if cls not in exclude_classes]
        top50  = [c for c in top100_classes[0:53] if c not in exclude_classes]

        # open the unambiguous affordances as a json file
        with open(image_root+f'/data_download/complete_dataset/metadata/top100_affordances.txt') as fin:
            top100_affordances = [line.strip() for line in fin]

        # open the descriptions as a json file
        with open(image_root+f'/data_download/complete_dataset/metadata/top100_descriptions.txt') as fin:
            top100_descriptions = [line.strip() for line in fin]

        # get the idx of exclude classes
        exclude_idx = [top100_classes.index(cls) for cls in exclude_classes]
        # remove the classes
        top100 = [top100_classes[idx] for idx in range(len(top100_classes)) if idx not in exclude_idx]
        top100_affordances = [top100_affordances[idx] for idx in range(len(top100_affordances)) if idx not in exclude_idx]
        top100_descriptions = [top100_descriptions[idx] for idx in range(len(top100_descriptions)) if idx not in exclude_idx]

        # just select the top 50 classes
        top100 = top100[0:50]
        top100_affordances = top100_affordances[0:50]
        top100_descriptions = top100_descriptions[0:50]

        # # generate a list of core concepts or train_classes
        # top100_idx = [23, 37, 25, 41, 22, 7, 48, 46, 9, 3]
        # longtail_idx = [1233, 1586, 1180, 1047, 1020, 585, 231, 343, 101, 843, 1332, 1050, 1022, 1331, 1188, 228, 1181, 393, 1416, 201, 1178, 525, 709, 745, 798, 496, 1356, 1188, 368, 367, 1313, 885, 1284, 1134, 1597, 496, 1278, 153, 980, 1304, 1346, 850, 969, 978, 1542, 879, 1050, 102, 134, 1273, 736, 629, 1592, 1461, 625, 1462, 1413, 1063, 1105, 978, 838, 853, 1268, 248, 1262, 373, 885, 1368, 757, 747, 1284, 601, 570, 827, 1048, 147, 296, 1512, 873, 1011, 1264, 1224, 1048, 945, 1, 654, 393, 1557, 216, 495, 1110, 3, 562, 721, 1445, 1545, 350, 120, 27, 1447]
        # affordance_idx = [12, 3, 32, 37, 8, 11, 14, 18, 2, 31]
        # ignore_top_50 = [cls for cls in target_classes if cls not in top50]

        # # define the training classes
        # train_classes = [top100_classes[idx] for idx in top100_idx]
        # # add the longtail classes
        # train_classes += [ignore_top_50[idx] for idx in longtail_idx]
        # # add values from the affordances
        # train_classes += [affordance_list[idx] for idx in affordance_idx]
        # print(train_classes)

        # use all the classess
        # train_classes = top100[1::n_core_concepts] + top100_affordances[0::n_core_concepts] + top100_descriptions[1::n_core_concepts]
        # select every 10th class for training
        # train_classes = train_classes[1::n_core_concepts]

        # use all the classes
        if use_affordances:
            # train_classes = top100_affordances[n_core_concepts::10] #+ top100_affordances[0::n_core_concepts] + top100_descriptions[1::n_core_concepts]
            train_classes = top100_affordances[0:48][n_core_concepts::8] #+ top100_affordances[0::n_core_concepts] + top100_descriptions[1::n_core_concepts]
        else:
            # train_classes = top100[n_core_concepts::10] #+ top100_affordances[0::n_core_concepts] + top100_descriptions[1::n_core_concepts]
            train_classes = top100[0:48][n_core_concepts::8] #+ top100_affordances[0::n_core_concepts] + top100_descriptions[1::n_core_concepts]

        print(train_classes)
        print(len(train_classes))

        # encode the train classes
        with torch.no_grad():
            caption_query = [f'An image of a {concept}' for concept in train_classes]
            caption_targets = clip_tokenizer(caption_query)
            caption_targets = caption_targets.to(device)
            core_feats = CLIP_Net.encode_text(caption_targets)
                            
        # This can be editted dynamically as we change the training approach etc..
        scene_objects = {}
        scene_queries = {}
        pseudo_objects = {}
        # list all files in the caption root
        instance_files = glob.glob(caption_root+f'/{dataset_split}/*.pkl')
        # iterate through each file
        for instance_file in tqdm.tqdm(instance_files):
            
            # # open the object info file
            # with h5py.File(instance_file, 'r') as f:
            #     try:
            #         # get the object feature data
            #         xfI = f['object_feats'][()]
            #         caption = f['captions'][()].decode('utf-8')
            #         scene_name = f['scene_name'][()].decode('utf-8')
            #         core_classes = f['core_classes'][()]
            #     except Exception as e:
            #         # print(e)
            #         continue

            scene_name = instance_file.split('/')[-1].split('.')[0]

            # if scene_name not in ['56a0ec536c', '8b5caf3398', '41b00feddb', '98b4ec142f', '7b6477cb95']:
            #     continue

            # open the pkl file
            with open(instance_file, 'rb') as f:
                segmented_objects = pickle.load(f)

            for obj_idx in range(len(segmented_objects['feature'])):
                xfI = segmented_objects['feature'][obj_idx]
                caption = segmented_objects['caption'][obj_idx]
                core_classes = segmented_objects['core_classes'][obj_idx]
                image_path = segmented_objects['image_path'][obj_idx]
                bbox = segmented_objects['bbox'][obj_idx][0]

                # check if this is a new scene
                if scene_name not in scene_objects.keys():
                    scene_objects[scene_name] = {
                        'object_feats': [],
                        'class_nums': [],
                        'captions': [],
                        'query_feats': [],
                        'unseen': []
                    }
                    scene_queries[scene_name] = {
                        'captions': [],
                        'query_feats': [],
                        'object_ids': [],
                        'unseen': []
                    }

                # if not core_classes:
                #     continue

                # check if the caption is invalid
                if caption == 'invalid' or caption == '':
                    continue

                captions = [caption]
                flag = [True]

                # append the object
                visual_features = np.expand_dims(xfI, axis=0)

                # append the object
                scene_objects[scene_name]['object_feats'].append(visual_features)
                scene_objects[scene_name]['captions'].append([])
                scene_objects[scene_name]['query_feats'].append([])
                scene_objects[scene_name]['unseen'].append([])
                object_idx = len(scene_objects[scene_name]['object_feats']) - 1
    
                # get the caption features
                with torch.no_grad():
                    caption_targets = clip_tokenizer(captions)
                    caption_targets = caption_targets.to(device)
                    caption_feats = CLIP_Net.encode_text(caption_targets)

                # iterate through each caption
                for i in range(len(captions)): 
                    caption = captions[i]  
                    xfT = caption_feats[i].unsqueeze(0).cpu().numpy() 
                    unseen = flag[i]

                    # append to the object
                    scene_objects[scene_name]['captions'][object_idx].append(caption)
                    scene_objects[scene_name]['query_feats'][object_idx].append(xfT)     
                    scene_objects[scene_name]['unseen'][object_idx].append(unseen)             

                    # check if caption not in the data structure
                    if caption not in scene_queries[scene_name]['captions']:
                        scene_queries[scene_name]['captions'].append(caption)
                        scene_queries[scene_name]['query_feats'].append(xfT)
                        scene_queries[scene_name]['object_ids'].append([str(object_idx)])
                        scene_queries[scene_name]['unseen'].append(unseen)
                    else:
                        scene_queries[scene_name]['object_ids'][scene_queries[scene_name]['captions'].index(caption)].append(str(object_idx))

        # query_feats: text feature of unique class in the scene
        # captions: captions of unique class in the scene for explainability
        # object_feats: visual feature of all unique object instances
        # query_labels: 0 or 1 for each object instance

        # create object data accessible in the get_item
        self.objects = {
            'object_feats': [],
            'captions': [],
            'query_feats': [],
            'unseen': []
        }
        for scene in scene_objects.keys():
            scene_object_feats = np.stack(scene_objects[scene]['object_feats'])
            # convert to torch 
            scene_object_feats = torch.from_numpy(scene_object_feats).to(device)
            scene_object_feats = scene_object_feats / scene_object_feats.norm(dim=-1, keepdim=True)

            # for each core concept
            for k in range(len(train_classes)):
                core_feat = core_feats[k]
                # normalise the core feeatures
                core_feats_norm = core_feat / core_feat.norm(dim=-1, keepdim=True)
                # calculate the cosine similarity
                sim = F.cosine_similarity(core_feats_norm, scene_object_feats, dim=-1).squeeze(1)
                # get the indices of the top 5 objects
                topk = torch.topk(sim, 5)
                # get the topk objects
                topk_objects = [scene_objects[scene]['object_feats'][idx] for idx in topk.indices]
                topk_captions = [scene_objects[scene]['captions'][idx] for idx in topk.indices]

                # print(train_classes[k], topk_captions)

                # for each object in the scene
                for obj_idx in range(len(topk_objects)):
                    self.objects['object_feats'].append(topk_objects[obj_idx])
                    self.objects['captions'].append([f'An image of a {train_classes[k]}'])
                    self.objects['query_feats'].append([core_feat.unsqueeze(0).cpu().numpy()])
                    self.objects['unseen'].append([False])

            # # for each object in the scene
            # for obj_idx in range(len(scene_objects[scene]['object_feats'])):
            #     self.objects['object_feats'].append(scene_objects[scene]['object_feats'][obj_idx])
            #     self.objects['captions'].append(scene_objects[scene]['captions'][obj_idx])
            #     self.objects['query_feats'].append(scene_objects[scene]['query_feats'][obj_idx])
            #     self.objects['unseen'].append(scene_objects[scene]['unseen'][obj_idx])

        # get the maximum number of objects in a single scene, to pad the object features
        self.max_objects = max([len(scene_objects[scene]['object_feats']) for scene in scene_objects.keys()])

        self.query_feats = []
        self.captions = []
        self.object_feats = []
        self.query_labels = []
        self.unseen = []
        self.n_objects = 0
        self.n_queries = 0
        for scene in scene_queries.keys():
            # check there is at least one object
            if not scene_objects[scene]['object_feats']:
                continue
            self.n_objects+=len(scene_objects[scene]['object_feats'])
            
            object_feats = scene_objects[scene]['object_feats']

            for i in range(len(scene_queries[scene]['query_feats'])):
                # append the query features
                self.query_feats.append(scene_queries[scene]['query_feats'][i])
                self.captions.append(scene_queries[scene]['captions'][i])
                # self.query_labels.append(scene_queries[scene]['object_ids'][i])
                self.query_labels.append('_'.join(scene_queries[scene]['object_ids'][i]))
                # append the object features
                self.object_feats.append(object_feats)
                # append the top100
                self.unseen.append(scene_queries[scene]['unseen'][i])
                self.n_queries+=1

        # for k in range(len(self.query_labels)):
        #     print(self.query_labels[k])

    def __getitem__(self, index):
        # get the object features
        object_feat = self.objects['object_feats'][index]
        # randomly select one of the captions and query features
        caption_idx = np.random.randint(len(self.objects['captions'][index]))
        caption = self.objects['captions'][index][caption_idx]
        query_feat = self.objects['query_feats'][index][caption_idx]
        unseen = self.objects['unseen'][index][caption_idx]

        object_feats = [object_feat]
        query_labels = '0'
        while len(object_feats) < 50:
            # randomly select an object
            random_int = np.random.randint(len(self.objects['object_feats']))
            # check caption is not in the object
            if caption in self.objects['captions'][random_int]:
                continue
            # add the feature to the object_feats
            object_feats.append(self.objects['object_feats'][random_int])

        # stack the object_feats
        object_feats = np.stack(object_feats).squeeze(1)

        # print(query_feat.shape)
        # print(query_labels)
        # print(caption)
        # print(object_feats.shape)
        # print(unseen)

        # return self.query_feats[index], query_labels, self.captions[index], object_feats, self.unseen[index]
        return query_feat, query_labels, caption, object_feats, unseen

    def __len__(self):
        # return len(self.captions)
        return len(self.objects['captions'])
    
class TopkImageClassification(Dataset):
    """CUB Captions Dataset.

    Args:
        image_root (string): Root directory where images are downloaded to.
        caption_root (string): Root directory where captions are downloaded to.
        target_classes (str or list): target class ids
            - if str, it is the name of the file with target classes (line by line)
            - if list, it is directly used to get classes
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        omit_ids (str, optional): Path of file with the list of image ids to omit,
            if not specified, use all images in the target classes.
        ids (str, optional): Path of file with the list of target image ids,
            if not specified, use all images in the target classes.
    """
    def __init__(self, image_root, caption_root,
                 split='val', pseudo_thresh=0.3, pseudo_method='scene', use_affordances=False, n_core_concepts=1, ntopk=5, n_negatives=100
                 ):

        print('topk_img_cls')
    
        # Initialize the CLIP model used in conceptgraphs
        device = 'cuda'
        CLIP_Net, _, clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", "laion2b_s32b_b79k"
        )
        CLIP_Net = CLIP_Net.to(device)
        CLIP_Net.eval()
        clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

        # get the dataset splits
        if split in ['train', 'val']:
            dataset_split = split
        else:
            dataset_split = 'train'
        
        splits_path = image_root+f'/data_download/complete_dataset/splits/nvs_sem_{dataset_split}.txt'
        # read each line and add to the splits file
        with open(splits_path) as fin:
            scenes = [line.strip() for line in fin]
        # sort the scnes for consistency
        scenes.sort()
        # print(scenes)

        # get the semantic classes
        semantic_classes = image_root+f'/data_download/complete_dataset/metadata/semantic_classes.txt'
        top100_classes = image_root+f'/data_download/complete_dataset/metadata/semantic_benchmark/top100.txt'
        with open(semantic_classes) as fin:
            semantic_classes = [line.strip() for line in fin]
        # # print(semantic_classes)

        with open(top100_classes) as fin:
            top100_classes = [line.strip() for line in fin]
        # # print(top100_classes)

        # remove some classes
        exclude_classes = ['ceiling', 'floor', 'wall']
        target_classes = [cls for cls in semantic_classes if cls not in exclude_classes]
        top50  = [c for c in top100_classes[0:53] if c not in exclude_classes]

        # open the unambiguous affordances as a json file
        with open(image_root+f'/data_download/complete_dataset/metadata/top100_affordances.txt') as fin:
            top100_affordances = [line.strip() for line in fin]

        # open the descriptions as a json file
        with open(image_root+f'/data_download/complete_dataset/metadata/top100_descriptions.txt') as fin:
            top100_descriptions = [line.strip() for line in fin]

        # open json file for captioned labels
        with open(image_root+f'/data_download/complete_dataset/metadata/captioned_labels.json') as fin:
            captioned_labels = json.load(fin)

        # get the idx of exclude classes
        exclude_idx = [top100_classes.index(cls) for cls in exclude_classes]
        # remove the classes
        top100 = [top100_classes[idx] for idx in range(len(top100_classes)) if idx not in exclude_idx]
        top100_affordances = [top100_affordances[idx] for idx in range(len(top100_affordances)) if idx not in exclude_idx]
        top100_descriptions = [top100_descriptions[idx] for idx in range(len(top100_descriptions)) if idx not in exclude_idx]

        # just select the top 50 classes
        top100 = top100[0:50]
        top100_affordances = top100_affordances[0:50]
        top100_descriptions = top100_descriptions[0:50]

        # # generate a list of core concepts or train_classes
        # top100_idx = [23, 37, 25, 41, 22, 7, 48, 46, 9, 3]
        # longtail_idx = [1233, 1586, 1180, 1047, 1020, 585, 231, 343, 101, 843, 1332, 1050, 1022, 1331, 1188, 228, 1181, 393, 1416, 201, 1178, 525, 709, 745, 798, 496, 1356, 1188, 368, 367, 1313, 885, 1284, 1134, 1597, 496, 1278, 153, 980, 1304, 1346, 850, 969, 978, 1542, 879, 1050, 102, 134, 1273, 736, 629, 1592, 1461, 625, 1462, 1413, 1063, 1105, 978, 838, 853, 1268, 248, 1262, 373, 885, 1368, 757, 747, 1284, 601, 570, 827, 1048, 147, 296, 1512, 873, 1011, 1264, 1224, 1048, 945, 1, 654, 393, 1557, 216, 495, 1110, 3, 562, 721, 1445, 1545, 350, 120, 27, 1447]
        # affordance_idx = [12, 3, 32, 37, 8, 11, 14, 18, 2, 31]
        # ignore_top_50 = [cls for cls in target_classes if cls not in top50]

        # # define the training classes
        # train_classes = [top100_classes[idx] for idx in top100_idx]
        # # add the longtail classes
        # train_classes += [ignore_top_50[idx] for idx in longtail_idx]
        # # add values from the affordances
        # train_classes += [affordance_list[idx] for idx in affordance_idx]
        # print(train_classes)

        # use all the classes
        if use_affordances:
            # train_classes = top100_affordances[n_core_concepts::10] #+ top100_affordances[0::n_core_concepts] + top100_descriptions[1::n_core_concepts]
            train_classes = top100_affordances[0:48][n_core_concepts::8] #+ top100_affordances[0::n_core_concepts] + top100_descriptions[1::n_core_concepts]
        else:
            # train_classes = top100[n_core_concepts::10] #+ top100_affordances[0::n_core_concepts] + top100_descriptions[1::n_core_concepts]
            train_classes = top100[0:48][n_core_concepts::8] #+ top100_affordances[0::n_core_concepts] + top100_descriptions[1::n_core_concepts]

        # get the captioned labels
        proposed_labels = train_classes.copy()
        # select the first 50
        count = 0
        for key, value in captioned_labels.items():
            count+=1
            if count > n_negatives:
                continue
            # check for intersection between value and train_classes
            if any(item in value for item in train_classes):
                print(key)
                continue
            proposed_labels.append(key)

        # select every 10th class for training
        # train_classes = train_classes[1::n_core_concepts]
        print(train_classes)
        print(proposed_labels)
        print(len(train_classes))
        self.train_classes = train_classes
        self.all_classes = top100 + top100_affordances + top100_descriptions
        self.all_objects = top100
        self.all_affordances = top100_affordances
        self.proposed_labels = proposed_labels

        # encode the train classes
        with torch.no_grad():
            caption_query = [f'An image of a {concept}' for concept in train_classes]
            caption_targets = clip_tokenizer(caption_query)
            caption_targets = caption_targets.to(device)
            core_feats = CLIP_Net.encode_text(caption_targets)
                            
        # This can be editted dynamically as we change the training approach etc..
        scene_objects = {}
        scene_queries = {}
        pseudo_objects = {}
        # list all files in the caption root
        instance_files = glob.glob(caption_root+f'/{dataset_split}/*.pkl')
        # iterate through each file
        for instance_file in tqdm.tqdm(instance_files):
            
            # # open the object info file
            # with h5py.File(instance_file, 'r') as f:
            #     try:
            #         # get the object feature data
            #         xfI = f['object_feats'][()]
            #         caption = f['captions'][()].decode('utf-8')
            #         scene_name = f['scene_name'][()].decode('utf-8')
            #         core_classes = f['core_classes'][()]
            #     except Exception as e:
            #         # print(e)
            #         continue

            scene_name = instance_file.split('/')[-1].split('.')[0]

            # if scene_name not in ['56a0ec536c', '8b5caf3398', '41b00feddb', '98b4ec142f', '7b6477cb95']:
            #     continue

            # open the pkl file
            with open(instance_file, 'rb') as f:
                segmented_objects = pickle.load(f)

            for obj_idx in range(len(segmented_objects['feature'])):
                xfI = segmented_objects['feature'][obj_idx]
                caption = segmented_objects['caption'][obj_idx]
                core_classes = segmented_objects['core_classes'][obj_idx]
                image_path = segmented_objects['image_path'][obj_idx]
                bbox = segmented_objects['bbox'][obj_idx][0]

                # check if this is a new scene
                if scene_name not in scene_objects.keys():
                    scene_objects[scene_name] = {
                        'object_feats': [],
                        'class_nums': [],
                        'captions': [],
                        'query_feats': [],
                        'unseen': []
                    }
                    scene_queries[scene_name] = {
                        'captions': [],
                        'query_feats': [],
                        'object_ids': [],
                        'unseen': []
                    }

                # if not core_classes:
                #     continue

                # check for invalid captions
                if caption == 'invalid' or caption == '':
                    continue

                # caption with core concepts
                captions = []
                flag = []
                for core in core_classes:
                    # check it is in the training classes
                    if core in train_classes:
                        captions.append(f'An image of a {core}')
                        flag.append(False)
                

                # if not captions:
                    # if pseudo_method == 'img_class_ueo':
                captions = ["An image of a "+train_classes[0]]
                flag = [False]
                    # else:
                        # continue

                        # print(core)
                        # # open the image
                        # image = Image.open(image_path)
                        # image = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                        # # display with matplotlib
                        # plt.figure()
                        # plt.imshow(image)
                        # plt.show()

                # # if empty, add the original object captions
                # if not captions:
                #     # otherwise label with caption
                #     captions = [caption]
                #     flag = [True]

                # append the object
                visual_features = np.expand_dims(xfI, axis=0)

                # append the object
                scene_objects[scene_name]['object_feats'].append(visual_features)
                scene_objects[scene_name]['captions'].append([])
                scene_objects[scene_name]['query_feats'].append([])
                scene_objects[scene_name]['unseen'].append([])
                object_idx = len(scene_objects[scene_name]['object_feats']) - 1
    
                # get the caption features
                with torch.no_grad():
                    caption_targets = clip_tokenizer(captions)
                    caption_targets = caption_targets.to(device)
                    caption_feats = CLIP_Net.encode_text(caption_targets)

                # iterate through each caption
                for i in range(len(captions)): 
                    caption = captions[i]  
                    xfT = caption_feats[i].unsqueeze(0).cpu().numpy() 
                    unseen = flag[i]

                    # append to the object
                    scene_objects[scene_name]['captions'][object_idx].append(caption)
                    scene_objects[scene_name]['query_feats'][object_idx].append(xfT)     
                    scene_objects[scene_name]['unseen'][object_idx].append(unseen)             

                    # check if caption not in the data structure
                    if caption not in scene_queries[scene_name]['captions']:
                        scene_queries[scene_name]['captions'].append(caption)
                        scene_queries[scene_name]['query_feats'].append(xfT)
                        scene_queries[scene_name]['object_ids'].append([str(object_idx)])
                        scene_queries[scene_name]['unseen'].append(unseen)
                    else:
                        scene_queries[scene_name]['object_ids'][scene_queries[scene_name]['captions'].index(caption)].append(str(object_idx))

        # query_feats: text feature of unique class in the scene
        # captions: captions of unique class in the scene for explainability
        # object_feats: visual feature of all unique object instances
        # query_labels: 0 or 1 for each object instance

        # create object data accessible in the get_item
        self.objects = {
            'object_feats': [],
            'captions': [],
            'query_feats': [],
            'unseen': []
        }
        for scene in scene_objects.keys():
            scene_object_feats = np.stack(scene_objects[scene]['object_feats'])
            # convert to torch 
            scene_object_feats = torch.from_numpy(scene_object_feats).to(device)
            scene_object_feats = scene_object_feats / scene_object_feats.norm(dim=-1, keepdim=True)

            # for each core concept
            for k in range(len(train_classes)):
                core_feat = core_feats[k]
                # normalise the core feeatures
                core_feats_norm = core_feat / core_feat.norm(dim=-1, keepdim=True)
                # calculate the cosine similarity
                sim = F.cosine_similarity(core_feats_norm, scene_object_feats, dim=-1).squeeze(1)
                # get the indices of the top 5 objects
                topk = torch.topk(sim, ntopk)
                # get the topk objects
                topk_objects = [scene_objects[scene]['object_feats'][idx] for idx in topk.indices]
                topk_captions = [scene_objects[scene]['captions'][idx] for idx in topk.indices]

                # print(train_classes[k], topk_captions)

                # for each object in the scene
                for obj_idx in range(len(topk_objects)):
                    self.objects['object_feats'].append(topk_objects[obj_idx])
                    self.objects['captions'].append([f'An image of a {train_classes[k]}'])
                    self.objects['query_feats'].append([core_feat.unsqueeze(0).cpu().numpy()])
                    self.objects['unseen'].append([False])

        # get the maximum number of objects in a single scene, to pad the object features
        self.max_objects = max([len(scene_objects[scene]['object_feats']) for scene in scene_objects.keys()])

        self.query_feats = []
        self.captions = []
        self.object_feats = []
        self.query_labels = []
        self.unseen = []
        self.n_objects = 0
        self.n_queries = 0
        for scene in scene_queries.keys():
            # check there is at least one object
            if not scene_objects[scene]['object_feats']:
                continue
            self.n_objects+=len(scene_objects[scene]['object_feats'])
            
            object_feats = scene_objects[scene]['object_feats']

            for i in range(len(scene_queries[scene]['query_feats'])):
                # append the query features
                self.query_feats.append(scene_queries[scene]['query_feats'][i])
                self.captions.append(scene_queries[scene]['captions'][i])
                # self.query_labels.append(scene_queries[scene]['object_ids'][i])
                self.query_labels.append('_'.join(scene_queries[scene]['object_ids'][i]))
                # append the object features
                self.object_feats.append(object_feats)
                # append the top100
                self.unseen.append(scene_queries[scene]['unseen'][i])
                self.n_queries+=1

        # for k in range(len(self.query_labels)):
        #     print(self.query_labels[k])

    def __getitem__(self, index):
        # get the object features
        object_feat = self.objects['object_feats'][index]
        # randomly select one of the captions and query features
        caption_idx = np.random.randint(len(self.objects['captions'][index]))
        caption = self.objects['captions'][index][caption_idx]
        query_feat = self.objects['query_feats'][index][caption_idx]
        unseen = self.objects['unseen'][index][caption_idx]
        query_labels = '0'

        # object_feats = [object_feat]
        # query_labels = '0'
        # while len(object_feats) < 50:
        #     # randomly select an object
        #     random_int = np.random.randint(len(self.objects['object_feats']))
        #     # check caption is not in the object
        #     if caption in self.objects['captions'][random_int]:
        #         continue
        #     # add the feature to the object_feats
        #     object_feats.append(self.objects['object_feats'][random_int])
                
        # object_feats = self.object_feats[index]
        # query_labels = self.query_labels[index].split('_')
        # query_caption = self.captions[index]
        # # get just the feats that are in the query labels
        # object_feats = [object_feats[int(idx)] for idx in query_labels]
        # # query feats
        # query_labels = '_'.join([str(idx) for idx in range(len(query_labels))])

        # # add other objects until we have 50
        # while len(object_feats) < 50:
        #     random_int = random.randint(0, len(self.object_feats)-1)
        #     random_caption = self.captions[random_int]
        #     # make sure we are getting an object with a different label
        #     if random_caption == query_caption:
        #         continue
        #     # select the first object from the query labels
        #     random_object = self.query_labels[random_int].split('_')[0]
        #     # get the object features
        #     random_feats = self.object_feats[random_int][int(random_object)]
        #     object_feats.append(random_feats)

        # stack the object_feats
        # object_feats = np.stack(object_feats).squeeze(1)

        # return self.query_feats[index], query_labels, self.captions[index], object_feats, self.unseen[index]
        return query_feat, query_labels, caption, object_feat, unseen
    
    def __len__(self):
        # return len(self.captions)
        return len(self.objects['captions'])
    
class TaskImageClassification(Dataset):
    """CUB Captions Dataset.

    Args:
        image_root (string): Root directory where images are downloaded to.
        caption_root (string): Root directory where captions are downloaded to.
        target_classes (str or list): target class ids
            - if str, it is the name of the file with target classes (line by line)
            - if list, it is directly used to get classes
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        omit_ids (str, optional): Path of file with the list of image ids to omit,
            if not specified, use all images in the target classes.
        ids (str, optional): Path of file with the list of target image ids,
            if not specified, use all images in the target classes.
    """
    def __init__(self, image_root, caption_root,
                 split='val', pseudo_thresh=0.3, pseudo_method='scene', use_affordances=False, n_core_concepts=1, ntopk=5, n_negatives=100
                 ):

        print('task_img_cls')
    
        # Initialize the CLIP model used in conceptgraphs
        device = 'cuda'
        CLIP_Net, _, clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", "laion2b_s32b_b79k"
        )
        CLIP_Net = CLIP_Net.to(device)
        CLIP_Net.eval()
        clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

        # get the dataset splits
        if split in ['train', 'val']:
            dataset_split = split
        else:
            dataset_split = 'train'
        
        splits_path = image_root+f'/data_download/complete_dataset/splits/nvs_sem_{dataset_split}.txt'
        # read each line and add to the splits file
        with open(splits_path) as fin:
            scenes = [line.strip() for line in fin]
        # sort the scnes for consistency
        scenes.sort()
        # print(scenes)

        # get the semantic classes
        semantic_classes = image_root+f'/data_download/complete_dataset/metadata/semantic_classes.txt'
        top100_classes = image_root+f'/data_download/complete_dataset/metadata/semantic_benchmark/top100.txt'
        with open(semantic_classes) as fin:
            semantic_classes = [line.strip() for line in fin]
        # # print(semantic_classes)

        with open(top100_classes) as fin:
            top100_classes = [line.strip() for line in fin]
        # # print(top100_classes)

        # remove some classes
        exclude_classes = ['ceiling', 'floor', 'wall']
        target_classes = [cls for cls in semantic_classes if cls not in exclude_classes]
        top50  = [c for c in top100_classes[0:53] if c not in exclude_classes]

        # open the unambiguous affordances as a json file
        with open(image_root+f'/data_download/complete_dataset/metadata/top100_affordances.txt') as fin:
            top100_affordances = [line.strip() for line in fin]

        # open the descriptions as a json file
        with open(image_root+f'/data_download/complete_dataset/metadata/top100_descriptions.txt') as fin:
            top100_descriptions = [line.strip() for line in fin]

        # open json file for captioned labels
        with open(image_root+f'/data_download/complete_dataset/metadata/captioned_task_labels.json') as fin:
            captioned_labels = json.load(fin)

        # get the idx of exclude classes
        exclude_idx = [top100_classes.index(cls) for cls in exclude_classes]
        # remove the classes
        top100 = [top100_classes[idx] for idx in range(len(top100_classes)) if idx not in exclude_idx]
        top100_affordances = [top100_affordances[idx] for idx in range(len(top100_affordances)) if idx not in exclude_idx]
        top100_descriptions = [top100_descriptions[idx] for idx in range(len(top100_descriptions)) if idx not in exclude_idx]

        # just select the top 50 classes
        top100 = top100[0:50]
        top100_affordances = top100_affordances[0:50]
        top100_descriptions = top100_descriptions[0:50]

        # # generate a list of core concepts or train_classes
        # top100_idx = [23, 37, 25, 41, 22, 7, 48, 46, 9, 3]
        # longtail_idx = [1233, 1586, 1180, 1047, 1020, 585, 231, 343, 101, 843, 1332, 1050, 1022, 1331, 1188, 228, 1181, 393, 1416, 201, 1178, 525, 709, 745, 798, 496, 1356, 1188, 368, 367, 1313, 885, 1284, 1134, 1597, 496, 1278, 153, 980, 1304, 1346, 850, 969, 978, 1542, 879, 1050, 102, 134, 1273, 736, 629, 1592, 1461, 625, 1462, 1413, 1063, 1105, 978, 838, 853, 1268, 248, 1262, 373, 885, 1368, 757, 747, 1284, 601, 570, 827, 1048, 147, 296, 1512, 873, 1011, 1264, 1224, 1048, 945, 1, 654, 393, 1557, 216, 495, 1110, 3, 562, 721, 1445, 1545, 350, 120, 27, 1447]
        # affordance_idx = [12, 3, 32, 37, 8, 11, 14, 18, 2, 31]
        # ignore_top_50 = [cls for cls in target_classes if cls not in top50]

        # # define the training classes
        # train_classes = [top100_classes[idx] for idx in top100_idx]
        # # add the longtail classes
        # train_classes += [ignore_top_50[idx] for idx in longtail_idx]
        # # add values from the affordances
        # train_classes += [affordance_list[idx] for idx in affordance_idx]
        # print(train_classes)

        # # use all the classes
        # if use_affordances:
        #     # train_classes = top100_affordances[n_core_concepts::10] #+ top100_affordances[0::n_core_concepts] + top100_descriptions[1::n_core_concepts]
        #     train_classes = top100_affordances[0:48][n_core_concepts::8] #+ top100_affordances[0::n_core_concepts] + top100_descriptions[1::n_core_concepts]
        # else:
        #     # train_classes = top100[n_core_concepts::10] #+ top100_affordances[0::n_core_concepts] + top100_descriptions[1::n_core_concepts]
        #     train_classes = top100[0:48][n_core_concepts::8] #+ top100_affordances[0::n_core_concepts] + top100_descriptions[1::n_core_concepts]

        
        # get the task set
        tasks = task_sets[n_core_concepts]
        train_classes = []
        for task in tasks:
            train_classes += task_objects[task]
        train_classes = list(set(train_classes))
        
        # get the captioned labels
        proposed_labels = train_classes.copy()
        # select the first 50
        count = 0
        for key, value in captioned_labels.items():
            count+=1
            if count > n_negatives:
                continue
            # check for intersection between value and train_classes
            if any(item in value for item in train_classes):
                print(key)
                continue
            proposed_labels.append(key)

        # select every 10th class for training
        # train_classes = train_classes[1::n_core_concepts]
        print(train_classes)
        print(proposed_labels)
        print(len(train_classes))
        self.train_classes = train_classes
        self.all_classes = top100 + top100_affordances + top100_descriptions
        self.all_objects = top100
        self.all_affordances = top100_affordances
        self.proposed_labels = proposed_labels

        # encode the train classes
        with torch.no_grad():
            caption_query = [f'An image of a {concept}' for concept in train_classes]
            caption_targets = clip_tokenizer(caption_query)
            caption_targets = caption_targets.to(device)
            core_feats = CLIP_Net.encode_text(caption_targets)
                            
        # This can be editted dynamically as we change the training approach etc..
        scene_objects = {}
        scene_queries = {}
        pseudo_objects = {}
        # list all files in the caption root
        instance_files = glob.glob(caption_root+f'/{dataset_split}/*.pkl')
        # iterate through each file
        for instance_file in tqdm.tqdm(instance_files):
            
            # # open the object info file
            # with h5py.File(instance_file, 'r') as f:
            #     try:
            #         # get the object feature data
            #         xfI = f['object_feats'][()]
            #         caption = f['captions'][()].decode('utf-8')
            #         scene_name = f['scene_name'][()].decode('utf-8')
            #         core_classes = f['core_classes'][()]
            #     except Exception as e:
            #         # print(e)
            #         continue

            scene_name = instance_file.split('/')[-1].split('.')[0]

            # if scene_name not in ['56a0ec536c', '8b5caf3398', '41b00feddb', '98b4ec142f', '7b6477cb95']:
            #     continue

            # open the pkl file
            with open(instance_file, 'rb') as f:
                segmented_objects = pickle.load(f)

            for obj_idx in range(len(segmented_objects['feature'])):
                xfI = segmented_objects['feature'][obj_idx]
                caption = segmented_objects['caption'][obj_idx]
                core_classes = segmented_objects['core_classes'][obj_idx]
                image_path = segmented_objects['image_path'][obj_idx]
                bbox = segmented_objects['bbox'][obj_idx][0]

                # check if this is a new scene
                if scene_name not in scene_objects.keys():
                    scene_objects[scene_name] = {
                        'object_feats': [],
                        'class_nums': [],
                        'captions': [],
                        'query_feats': [],
                        'unseen': []
                    }
                    scene_queries[scene_name] = {
                        'captions': [],
                        'query_feats': [],
                        'object_ids': [],
                        'unseen': []
                    }

                # if not core_classes:
                #     continue

                # check for invalid captions
                if caption == 'invalid' or caption == '':
                    continue

                # caption with core concepts
                captions = []
                flag = []
                for core in core_classes:
                    # check it is in the training classes
                    if core in train_classes:
                        captions.append(f'An image of a {core}')
                        flag.append(False)
                

                # if not captions:
                    # if pseudo_method == 'img_class_ueo':
                captions = ["An image of a "+train_classes[0]]
                flag = [False]
                    # else:
                        # continue

                        # print(core)
                        # # open the image
                        # image = Image.open(image_path)
                        # image = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                        # # display with matplotlib
                        # plt.figure()
                        # plt.imshow(image)
                        # plt.show()

                # # if empty, add the original object captions
                # if not captions:
                #     # otherwise label with caption
                #     captions = [caption]
                #     flag = [True]

                # append the object
                visual_features = np.expand_dims(xfI, axis=0)

                # append the object
                scene_objects[scene_name]['object_feats'].append(visual_features)
                scene_objects[scene_name]['captions'].append([])
                scene_objects[scene_name]['query_feats'].append([])
                scene_objects[scene_name]['unseen'].append([])
                object_idx = len(scene_objects[scene_name]['object_feats']) - 1
    
                # get the caption features
                with torch.no_grad():
                    caption_targets = clip_tokenizer(captions)
                    caption_targets = caption_targets.to(device)
                    caption_feats = CLIP_Net.encode_text(caption_targets)

                # iterate through each caption
                for i in range(len(captions)): 
                    caption = captions[i]  
                    xfT = caption_feats[i].unsqueeze(0).cpu().numpy() 
                    unseen = flag[i]

                    # append to the object
                    scene_objects[scene_name]['captions'][object_idx].append(caption)
                    scene_objects[scene_name]['query_feats'][object_idx].append(xfT)     
                    scene_objects[scene_name]['unseen'][object_idx].append(unseen)             

                    # check if caption not in the data structure
                    if caption not in scene_queries[scene_name]['captions']:
                        scene_queries[scene_name]['captions'].append(caption)
                        scene_queries[scene_name]['query_feats'].append(xfT)
                        scene_queries[scene_name]['object_ids'].append([str(object_idx)])
                        scene_queries[scene_name]['unseen'].append(unseen)
                    else:
                        scene_queries[scene_name]['object_ids'][scene_queries[scene_name]['captions'].index(caption)].append(str(object_idx))

        # query_feats: text feature of unique class in the scene
        # captions: captions of unique class in the scene for explainability
        # object_feats: visual feature of all unique object instances
        # query_labels: 0 or 1 for each object instance

        # create object data accessible in the get_item
        self.objects = {
            'object_feats': [],
            'captions': [],
            'query_feats': [],
            'unseen': []
        }
        for scene in scene_objects.keys():
            scene_object_feats = np.stack(scene_objects[scene]['object_feats'])
            # convert to torch 
            scene_object_feats = torch.from_numpy(scene_object_feats).to(device)
            scene_object_feats = scene_object_feats / scene_object_feats.norm(dim=-1, keepdim=True)

            # for each core concept
            for k in range(len(train_classes)):
                core_feat = core_feats[k]
                # normalise the core feeatures
                core_feats_norm = core_feat / core_feat.norm(dim=-1, keepdim=True)
                # calculate the cosine similarity
                sim = F.cosine_similarity(core_feats_norm, scene_object_feats, dim=-1).squeeze(1)
                # get the indices of the top 5 objects
                topk = torch.topk(sim, ntopk)
                # get the topk objects
                topk_objects = [scene_objects[scene]['object_feats'][idx] for idx in topk.indices]
                topk_captions = [scene_objects[scene]['captions'][idx] for idx in topk.indices]

                # print(train_classes[k], topk_captions)

                # for each object in the scene
                for obj_idx in range(len(topk_objects)):
                    self.objects['object_feats'].append(topk_objects[obj_idx])
                    self.objects['captions'].append([f'An image of a {train_classes[k]}'])
                    self.objects['query_feats'].append([core_feat.unsqueeze(0).cpu().numpy()])
                    self.objects['unseen'].append([False])

        # get the maximum number of objects in a single scene, to pad the object features
        self.max_objects = max([len(scene_objects[scene]['object_feats']) for scene in scene_objects.keys()])

        self.query_feats = []
        self.captions = []
        self.object_feats = []
        self.query_labels = []
        self.unseen = []
        self.n_objects = 0
        self.n_queries = 0
        for scene in scene_queries.keys():
            # check there is at least one object
            if not scene_objects[scene]['object_feats']:
                continue
            self.n_objects+=len(scene_objects[scene]['object_feats'])
            
            object_feats = scene_objects[scene]['object_feats']

            for i in range(len(scene_queries[scene]['query_feats'])):
                # append the query features
                self.query_feats.append(scene_queries[scene]['query_feats'][i])
                self.captions.append(scene_queries[scene]['captions'][i])
                # self.query_labels.append(scene_queries[scene]['object_ids'][i])
                self.query_labels.append('_'.join(scene_queries[scene]['object_ids'][i]))
                # append the object features
                self.object_feats.append(object_feats)
                # append the top100
                self.unseen.append(scene_queries[scene]['unseen'][i])
                self.n_queries+=1

        # for k in range(len(self.query_labels)):
        #     print(self.query_labels[k])

    def __getitem__(self, index):
        # get the object features
        object_feat = self.objects['object_feats'][index]
        # randomly select one of the captions and query features
        caption_idx = np.random.randint(len(self.objects['captions'][index]))
        caption = self.objects['captions'][index][caption_idx]
        query_feat = self.objects['query_feats'][index][caption_idx]
        unseen = self.objects['unseen'][index][caption_idx]
        query_labels = '0'

        # object_feats = [object_feat]
        # query_labels = '0'
        # while len(object_feats) < 50:
        #     # randomly select an object
        #     random_int = np.random.randint(len(self.objects['object_feats']))
        #     # check caption is not in the object
        #     if caption in self.objects['captions'][random_int]:
        #         continue
        #     # add the feature to the object_feats
        #     object_feats.append(self.objects['object_feats'][random_int])
                
        # object_feats = self.object_feats[index]
        # query_labels = self.query_labels[index].split('_')
        # query_caption = self.captions[index]
        # # get just the feats that are in the query labels
        # object_feats = [object_feats[int(idx)] for idx in query_labels]
        # # query feats
        # query_labels = '_'.join([str(idx) for idx in range(len(query_labels))])

        # # add other objects until we have 50
        # while len(object_feats) < 50:
        #     random_int = random.randint(0, len(self.object_feats)-1)
        #     random_caption = self.captions[random_int]
        #     # make sure we are getting an object with a different label
        #     if random_caption == query_caption:
        #         continue
        #     # select the first object from the query labels
        #     random_object = self.query_labels[random_int].split('_')[0]
        #     # get the object features
        #     random_feats = self.object_feats[random_int][int(random_object)]
        #     object_feats.append(random_feats)

        # stack the object_feats
        # object_feats = np.stack(object_feats).squeeze(1)

        # return self.query_feats[index], query_labels, self.captions[index], object_feats, self.unseen[index]
        return query_feat, query_labels, caption, object_feat, unseen
    
    def __len__(self):
        # return len(self.captions)
        return len(self.objects['captions'])
    
class SegmentImageClassification(Dataset):
    """CUB Captions Dataset.

    Args:
        image_root (string): Root directory where images are downloaded to.
        caption_root (string): Root directory where captions are downloaded to.
        target_classes (str or list): target class ids
            - if str, it is the name of the file with target classes (line by line)
            - if list, it is directly used to get classes
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        omit_ids (str, optional): Path of file with the list of image ids to omit,
            if not specified, use all images in the target classes.
        ids (str, optional): Path of file with the list of target image ids,
            if not specified, use all images in the target classes.
    """
    def __init__(self, image_root, caption_root,
                 split='val', pseudo_thresh=0.3, pseudo_method='scene', use_affordances=False, n_core_concepts=1
                 ):

        print('image classification')
    
        # Initialize the CLIP model used in conceptgraphs
        device = 'cuda'
        CLIP_Net, _, clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", "laion2b_s32b_b79k"
        )
        CLIP_Net = CLIP_Net.to(device)
        CLIP_Net.eval()
        clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

        # get the dataset splits
        if split in ['train', 'val']:
            dataset_split = split
        else:
            dataset_split = 'train'
        
        splits_path = image_root+f'/data_download/complete_dataset/splits/nvs_sem_{dataset_split}.txt'
        # read each line and add to the splits file
        with open(splits_path) as fin:
            scenes = [line.strip() for line in fin]
        # sort the scnes for consistency
        scenes.sort()
        # print(scenes)

        # get the semantic classes
        semantic_classes = image_root+f'/data_download/complete_dataset/metadata/semantic_classes.txt'
        top100_classes = image_root+f'/data_download/complete_dataset/metadata/semantic_benchmark/top100.txt'
        with open(semantic_classes) as fin:
            semantic_classes = [line.strip() for line in fin]
        # # print(semantic_classes)

        with open(top100_classes) as fin:
            top100_classes = [line.strip() for line in fin]
        # # print(top100_classes)

        # remove some classes
        exclude_classes = ['ceiling', 'floor', 'wall']
        target_classes = [cls for cls in semantic_classes if cls not in exclude_classes]
        top50  = [c for c in top100_classes[0:53] if c not in exclude_classes]

        # open the unambiguous affordances as a json file
        with open(image_root+f'/data_download/complete_dataset/metadata/top100_affordances.txt') as fin:
            top100_affordances = [line.strip() for line in fin]

        # open the descriptions as a json file
        with open(image_root+f'/data_download/complete_dataset/metadata/top100_descriptions.txt') as fin:
            top100_descriptions = [line.strip() for line in fin]

        # open json file for captioned labels
        with open(image_root+f'/data_download/complete_dataset/metadata/captioned_labels.json') as fin:
            captioned_labels = json.load(fin)

        # get the idx of exclude classes
        exclude_idx = [top100_classes.index(cls) for cls in exclude_classes]
        # remove the classes
        top100 = [top100_classes[idx] for idx in range(len(top100_classes)) if idx not in exclude_idx]
        top100_affordances = [top100_affordances[idx] for idx in range(len(top100_affordances)) if idx not in exclude_idx]
        top100_descriptions = [top100_descriptions[idx] for idx in range(len(top100_descriptions)) if idx not in exclude_idx]

        # just select the top 50 classes
        top100 = top100[0:50]
        top100_affordances = top100_affordances[0:50]
        top100_descriptions = top100_descriptions[0:50]

        # # generate a list of core concepts or train_classes
        # top100_idx = [23, 37, 25, 41, 22, 7, 48, 46, 9, 3]
        # longtail_idx = [1233, 1586, 1180, 1047, 1020, 585, 231, 343, 101, 843, 1332, 1050, 1022, 1331, 1188, 228, 1181, 393, 1416, 201, 1178, 525, 709, 745, 798, 496, 1356, 1188, 368, 367, 1313, 885, 1284, 1134, 1597, 496, 1278, 153, 980, 1304, 1346, 850, 969, 978, 1542, 879, 1050, 102, 134, 1273, 736, 629, 1592, 1461, 625, 1462, 1413, 1063, 1105, 978, 838, 853, 1268, 248, 1262, 373, 885, 1368, 757, 747, 1284, 601, 570, 827, 1048, 147, 296, 1512, 873, 1011, 1264, 1224, 1048, 945, 1, 654, 393, 1557, 216, 495, 1110, 3, 562, 721, 1445, 1545, 350, 120, 27, 1447]
        # affordance_idx = [12, 3, 32, 37, 8, 11, 14, 18, 2, 31]
        # ignore_top_50 = [cls for cls in target_classes if cls not in top50]

        # # define the training classes
        # train_classes = [top100_classes[idx] for idx in top100_idx]
        # # add the longtail classes
        # train_classes += [ignore_top_50[idx] for idx in longtail_idx]
        # # add values from the affordances
        # train_classes += [affordance_list[idx] for idx in affordance_idx]
        # print(train_classes)

        # use all the classes
        if use_affordances:
            # train_classes = top100_affordances[n_core_concepts::10] #+ top100_affordances[0::n_core_concepts] + top100_descriptions[1::n_core_concepts]
            train_classes = top100_affordances[0:48][n_core_concepts::8] #+ top100_affordances[0::n_core_concepts] + top100_descriptions[1::n_core_concepts]
        else:
            # train_classes = top100[n_core_concepts::10] #+ top100_affordances[0::n_core_concepts] + top100_descriptions[1::n_core_concepts]
            train_classes = top100[0:48][n_core_concepts::8] #+ top100_affordances[0::n_core_concepts] + top100_descriptions[1::n_core_concepts]

        # get the captioned labels
        proposed_labels = train_classes.copy()
        # select the first 50
        count = 0
        for key, value in captioned_labels.items():
            count+=1
            if count > 100:
                continue
            # check for intersection between value and train_classes
            if any(item in value for item in train_classes):
                print(key)
                continue
            proposed_labels.append(key)

        # select every 10th class for training
        # train_classes = train_classes[1::n_core_concepts]
        print(train_classes)
        print(proposed_labels)
        print(len(train_classes))
        self.train_classes = train_classes
        self.all_classes = top100 + top100_affordances + top100_descriptions
        self.all_objects = top100
        self.all_affordances = top100_affordances
        self.proposed_labels = proposed_labels

        # encode the train classes
        with torch.no_grad():
            caption_query = [f'An image of a {concept}' for concept in train_classes]
            caption_targets = clip_tokenizer(caption_query)
            caption_targets = caption_targets.to(device)
            core_feats = CLIP_Net.encode_text(caption_targets)
                            
        # This can be editted dynamically as we change the training approach etc..
        scene_objects = {}
        scene_queries = {}
        pseudo_objects = {}
        # list all files in the caption root
        instance_files = glob.glob(caption_root+f'/{dataset_split}/*.pkl')
        # iterate through each file
        for instance_file in tqdm.tqdm(instance_files):
            
            # # open the object info file
            # with h5py.File(instance_file, 'r') as f:
            #     try:
            #         # get the object feature data
            #         xfI = f['object_feats'][()]
            #         caption = f['captions'][()].decode('utf-8')
            #         scene_name = f['scene_name'][()].decode('utf-8')
            #         core_classes = f['core_classes'][()]
            #     except Exception as e:
            #         # print(e)
            #         continue

            scene_name = instance_file.split('/')[-1].split('.')[0]

            # if scene_name not in ['56a0ec536c', '8b5caf3398', '41b00feddb', '98b4ec142f', '7b6477cb95']:
            #     continue

            # open the pkl file
            with open(instance_file, 'rb') as f:
                segmented_objects = pickle.load(f)

            for obj_idx in range(len(segmented_objects['feature'])):
                xfI = segmented_objects['feature'][obj_idx]
                caption = segmented_objects['caption'][obj_idx]
                core_classes = segmented_objects['core_classes'][obj_idx]
                image_path = segmented_objects['image_path'][obj_idx]
                bbox = segmented_objects['bbox'][obj_idx][0]

                # check if this is a new scene
                if scene_name not in scene_objects.keys():
                    scene_objects[scene_name] = {
                        'object_feats': [],
                        'class_nums': [],
                        'captions': [],
                        'query_feats': [],
                        'unseen': []
                    }
                    scene_queries[scene_name] = {
                        'captions': [],
                        'query_feats': [],
                        'object_ids': [],
                        'unseen': []
                    }

                # if not core_classes:
                #     continue

                # check for invalid captions
                if caption == 'invalid' or caption == '':
                    continue

                # caption with core concepts
                if pseudo_method in ['img_class_ours', 'img_class_ueo']:
                    captions = []
                    flag = []
                    for core in core_classes:
                        # check it is in the training classes
                        if core in train_classes:
                            captions.append(f'An image of a {core}')
                            flag.append(False)

                # if pseudo_method is cosine sim
                elif pseudo_method == 'img_class_cosine':
                    # normalise the core feeatures
                    core_feats_norm = core_feats / core_feats.norm(dim=-1, keepdim=True)
                    # normalise the object features
                    object_feats_norm = xfI / xfI.norm(dim=-1, keepdim=True)
                    # put tensors on device
                    object_feats_norm = object_feats_norm.to(device)
                    # calculate the cosine similarity
                    sim = F.cosine_similarity(core_feats_norm, object_feats_norm, dim=-1)
                    # get all the concepts with a similarity above the threshold
                    above_thresh = sim > pseudo_thresh
                    # get all concepts above the threshold
                    core_concepts = [train_classes[idx] for idx in range(len(train_classes)) if above_thresh[idx]]

                    captions = []
                    flag = []
                    for concept in core_concepts:
                        captions.append(f'An image of a {concept}')
                        flag.append(False)
                

                if not captions:
                    # continue
                    if pseudo_method == 'img_class_ueo':
                        captions = ["An image of a "+train_classes[0]]
                        flag = [False]
                    else:
                        continue

                        # print(core)
                        # # open the image
                        # image = Image.open(image_path)
                        # image = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                        # # display with matplotlib
                        # plt.figure()
                        # plt.imshow(image)
                        # plt.show()

                # # if empty, add the original object captions
                # if not captions:
                #     # otherwise label with caption
                #     captions = [caption]
                #     flag = [True]

                # append the object
                visual_features = np.expand_dims(xfI, axis=0)

                # append the object
                scene_objects[scene_name]['object_feats'].append(visual_features)
                scene_objects[scene_name]['captions'].append([])
                scene_objects[scene_name]['query_feats'].append([])
                scene_objects[scene_name]['unseen'].append([])
                object_idx = len(scene_objects[scene_name]['object_feats']) - 1
    
                # get the caption features
                with torch.no_grad():
                    caption_targets = clip_tokenizer(captions)
                    caption_targets = caption_targets.to(device)
                    caption_feats = CLIP_Net.encode_text(caption_targets)

                # iterate through each caption
                for i in range(len(captions)): 
                    caption = captions[i]  
                    xfT = caption_feats[i].unsqueeze(0).cpu().numpy() 
                    unseen = flag[i]

                    # append to the object
                    scene_objects[scene_name]['captions'][object_idx].append(caption)
                    scene_objects[scene_name]['query_feats'][object_idx].append(xfT)     
                    scene_objects[scene_name]['unseen'][object_idx].append(unseen)             

                    # check if caption not in the data structure
                    if caption not in scene_queries[scene_name]['captions']:
                        scene_queries[scene_name]['captions'].append(caption)
                        scene_queries[scene_name]['query_feats'].append(xfT)
                        scene_queries[scene_name]['object_ids'].append([str(object_idx)])
                        scene_queries[scene_name]['unseen'].append(unseen)
                    else:
                        scene_queries[scene_name]['object_ids'][scene_queries[scene_name]['captions'].index(caption)].append(str(object_idx))

        # query_feats: text feature of unique class in the scene
        # captions: captions of unique class in the scene for explainability
        # object_feats: visual feature of all unique object instances
        # query_labels: 0 or 1 for each object instance

        # create object data accessible in the get_item
        self.objects = {
            'object_feats': [],
            'captions': [],
            'query_feats': [],
            'unseen': []
        }
        for scene in scene_objects.keys():
            # for each object in the scene
            for obj_idx in range(len(scene_objects[scene]['object_feats'])):
                self.objects['object_feats'].append(scene_objects[scene]['object_feats'][obj_idx])
                self.objects['captions'].append(scene_objects[scene]['captions'][obj_idx])
                self.objects['query_feats'].append(scene_objects[scene]['query_feats'][obj_idx])
                self.objects['unseen'].append(scene_objects[scene]['unseen'][obj_idx])

        # get the maximum number of objects in a single scene, to pad the object features
        self.max_objects = max([len(scene_objects[scene]['object_feats']) for scene in scene_objects.keys()])

        self.query_feats = []
        self.captions = []
        self.object_feats = []
        self.query_labels = []
        self.unseen = []
        self.n_objects = 0
        self.n_queries = 0
        for scene in scene_queries.keys():
            # check there is at least one object
            if not scene_objects[scene]['object_feats']:
                continue
            self.n_objects+=len(scene_objects[scene]['object_feats'])
            
            object_feats = scene_objects[scene]['object_feats']

            for i in range(len(scene_queries[scene]['query_feats'])):
                # append the query features
                self.query_feats.append(scene_queries[scene]['query_feats'][i])
                self.captions.append(scene_queries[scene]['captions'][i])
                # self.query_labels.append(scene_queries[scene]['object_ids'][i])
                self.query_labels.append('_'.join(scene_queries[scene]['object_ids'][i]))
                # append the object features
                self.object_feats.append(object_feats)
                # append the top100
                self.unseen.append(scene_queries[scene]['unseen'][i])
                self.n_queries+=1

        # for k in range(len(self.query_labels)):
        #     print(self.query_labels[k])

    def __getitem__(self, index):
        # get the object features
        object_feat = self.objects['object_feats'][index]
        # randomly select one of the captions and query features
        caption_idx = np.random.randint(len(self.objects['captions'][index]))
        caption = self.objects['captions'][index][caption_idx]
        query_feat = self.objects['query_feats'][index][caption_idx]
        unseen = self.objects['unseen'][index][caption_idx]
        query_labels = '0'

        # object_feats = [object_feat]
        # query_labels = '0'
        # while len(object_feats) < 50:
        #     # randomly select an object
        #     random_int = np.random.randint(len(self.objects['object_feats']))
        #     # check caption is not in the object
        #     if caption in self.objects['captions'][random_int]:
        #         continue
        #     # add the feature to the object_feats
        #     object_feats.append(self.objects['object_feats'][random_int])
                
        # object_feats = self.object_feats[index]
        # query_labels = self.query_labels[index].split('_')
        # query_caption = self.captions[index]
        # # get just the feats that are in the query labels
        # object_feats = [object_feats[int(idx)] for idx in query_labels]
        # # query feats
        # query_labels = '_'.join([str(idx) for idx in range(len(query_labels))])

        # # add other objects until we have 50
        # while len(object_feats) < 50:
        #     random_int = random.randint(0, len(self.object_feats)-1)
        #     random_caption = self.captions[random_int]
        #     # make sure we are getting an object with a different label
        #     if random_caption == query_caption:
        #         continue
        #     # select the first object from the query labels
        #     random_object = self.query_labels[random_int].split('_')[0]
        #     # get the object features
        #     random_feats = self.object_feats[random_int][int(random_object)]
        #     object_feats.append(random_feats)

        # stack the object_feats
        # object_feats = np.stack(object_feats).squeeze(1)

        # return self.query_feats[index], query_labels, self.captions[index], object_feats, self.unseen[index]
        return query_feat, query_labels, caption, object_feat, unseen
    
    def __len__(self):
        # return len(self.captions)
        return len(self.objects['captions'])