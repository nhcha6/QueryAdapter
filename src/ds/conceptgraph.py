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

GENERATE_EVAL_DATA = False

if GENERATE_EVAL_DATA:
    import open3d as o3d
    from conceptgraph.utils.ious import *
    from conceptgraph.utils.eval import compute_pred_gt_associations

def pad_text(num):
    if num<10:
        return '0000'+str(num)
    if num<100:
        return '000'+str(num)
           
    if num<1000:
        return '00'+str(num)

class ConceptgraphSampler(Sampler):
    """ Sampler for CUB Captions training.

    Args:
        dataset (CUBCaption object): dataset object to apply the sampler.
        batch_size (int): batch size.
        adjust_epoch (bool): if true, the iterations for one epoch is re-calculated.
    """
    def __init__(self, dataset, batch_size, adjust_epoch=True):
        self.dataset = dataset
        self.batch_size = batch_size
        print("Batch:",self.batch_size)
        self.target_classes = dataset.target_classes
        if batch_size != len(self.target_classes):
            raise ValueError(f'{batch_size} != {len(self.target_classes)}')
        self.index_to_class = dataset.index_to_class
        self.class_to_indices = dataset.class_to_indices
        self.n_items = len(self.index_to_class)

        if adjust_epoch:
            self.n_iters = int(self.n_items / len(self.target_classes))
        else:
            self.n_iters = self.n_items

    def __iter__(self):
        batch = []
        indices = list(range(self.n_items))

        np.random.shuffle(indices)
        for cur_iter, idx in enumerate(indices):
            batch = [idx]
            pos_cls = self.index_to_class[idx]
            for cls_num, _indices in self.class_to_indices.items():
                if cls_num == pos_cls:
                    continue
                else:
                    batch.append(np.random.choice(_indices))
            np.random.shuffle(batch)
            if cur_iter > self.n_iters:
                return
            yield batch


    def __len__(self):
        return self.n_iters
    
class ConceptgraphSceneQuery(Dataset):
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
                 target_classes, split='val',
                 transform=None, target_transform=None, seen_classes='top100', use_affordances = False, n_core_concepts=1
                 ):

        # self.image_root = os.path.expanduser(image_root)
        # self.caption_root = os.path.expanduser(caption_root)
        # device = 'cuda'
        # CLIP_Net = load_model(device='cuda', model_path=None)
        # CLIP_Net.eval()

        device = 'cuda'
        CLIP_Net, _, clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", "laion2b_s32b_b79k"
        )
        CLIP_Net = CLIP_Net.to(device)
        CLIP_Net.eval()
        clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

        # if isinstance(target_classes, str):
        #     with open(target_classes) as fin:
        #         _classes = [int(line.strip().split('_')[1]) - 1 for line in fin]
        #     target_classes = _classes
        

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
        # print(semantic_classes)

        with open(top100_classes) as fin:
            top100_classes = [line.strip() for line in fin]
        # print(top100_classes)
        
        # remove some classes
        exclude_classes = ['ceiling', 'floor', 'wall', 'object']
        target_classes = [cls for cls in semantic_classes if cls not in exclude_classes]
        self.target_classes = target_classes
        top50  = [c for c in top100_classes[0:53] if c not in exclude_classes]
        top100 = [c for c in top100_classes if c not in exclude_classes]

        # open the unambiguous affordances as a json file
        with open(image_root+f'/data_download/complete_dataset/metadata/unambiguous_affordances.json') as fin:
            unambiguous_affordances = json.load(fin)

        affordance_features = {}
        for affordance, object_top100 in unambiguous_affordances.items():
            with torch.no_grad():
                caption_query = [f'an image of {affordance}']
                caption_targets = clip_tokenizer(caption_query)
                caption_targets = caption_targets.to(device)
                caption_feat = CLIP_Net.encode_text(caption_targets)
                affordance_features[object_top100] = caption_feat
        unambiguous_affordances = {v: k for k, v in unambiguous_affordances.items()}
        affordance_list = list(unambiguous_affordances.values())
        # sort the list for consistency
        affordance_list.sort()

        # # generate a list of core concepts or train_classes
        # # 10 random indices from the top100 classes
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

        # just use the top50
        affordance_list = affordance_list[0:50]

        # use all the classes
        target_classes = top100
        # select every 10th class for training
        train_classes = target_classes[1::n_core_concepts]
        print(train_classes)

        # # define the training and testing classses
        # if seen_classes == 'top100':
        #     train_classes = top100_classes
        #     test_classes = semantic_classes
        #     calib_classes = top100_classes
        # elif seen_classes == 'half':
        #     train_classes = semantic_classes[0::2]
        #     test_classes = semantic_classes
        #     calib_classes = semantic_classes[0::2]
        # elif seen_classes == 'top100_half':
        #     train_classes = top100_classes[0:53]
        #     test_classes = semantic_classes
        #     calib_classes = top100_classes[0:53]

        # # define the target classes
        # if split == 'train':
        #     # target_classes = top100_classes
        #     target_classes = train_classes
        # elif split == 'calib':
        #     target_classes = calib_classes
        #     # scenes = scenes[-50:]
        # else:
        #     target_classes = test_classes

        ###################### PREPARE THE DATASET FOR TRAINING #####################

        # This can be editted dynamically as we change the training approach etc..

        scene_objects = {}
        scene_gt_queries = {}
        # list all files in the caption root
        instance_files = glob.glob(caption_root+f'/{dataset_split}/*.h5')
        # iterate through each file
        for instance_file in tqdm.tqdm(instance_files):
            scene_name = instance_file.split('/')[-1].split('_')[0]

            # if scene_name not in target_scenes:
            #     continue

            if scene_name not in ['56a0ec536c', '8b5caf3398', '41b00feddb', '98b4ec142f', '7b6477cb95']:
                continue

            with open(image_root+f'/data_download/complete_dataset/data/{scene_name}/scans/segments_anno.json') as fin:
                segments = json.load(fin)

            # initialise the data structures for the scene
            if scene_name not in scene_objects.keys():
                # dictionary to store the object features
                scene_objects[scene_name] = {
                    'object_feats': [],
                    'object_cls': [],
                    'object_file': []
                }

                # to store info about the gt objects
                scene_gt_queries[scene_name] = {}

                # # open the segments metadata
                # with open(image_root+f'/data_download/complete_dataset/data/{scene_name}/scans/segments_anno.json') as fin:
                #     segments = json.load(fin)
                #     scene_gt_queries[scene_name] = set()
                #     for gt_obj in segments['segGroups']:
                #         if gt_obj['label'] in self.target_classes:
                #             scene_gt_queries[scene_name].add(gt_obj['label'])

                # get the gt_boxes
                if GENERATE_EVAL_DATA:
                    gt_boxes, gt_labels, gt_pcd, pcd_disp, gt_annos = self.get_gt_boxes(image_root, scene_name, semantic_classes)
                    scene_gt_queries[scene_name]['gt_labels'] = gt_labels
                    scene_gt_queries[scene_name]['gt_boxes'] = gt_boxes
                    scene_gt_queries[scene_name]['gt_annos'] = gt_annos
                    scene_gt_queries[scene_name]['gt_pcd'] = gt_pcd
                    scene_gt_queries[scene_name]['pcd_disp'] = pcd_disp
                # get the labels of objects that are visible in the scene
                gt_labels = self.get_gt_labels(image_root, scene_name, target_classes)
                scene_gt_queries[scene_name]['gt_labels'] = gt_labels
            
            # open the object info file
            with h5py.File(instance_file, 'r') as f:
                try:
                    # get the object feature data
                    image_names = f['image_names']
                    crops = f['crops']
                    visual_features = f['visual_features'][()]
                    xfI = f['object_feat'][()]
                    gt_class = f['gt_class'][()].decode('utf-8')
                    flag = [False]
                    captions = [f'an image of a {gt_class}']
                    pcd = f['pcd'][()]

                    # get the caption data
                    init_captions = f['captions']
                    object_tag = f['object_tags'][()].decode('utf-8')
                    object_description = f['object_description'][()].decode('utf-8')
                    object_affordance = f['object_affordance'][()].decode('utf-8')
                except Exception as e:
                    # print(e)
                    continue

                # # convert the pcd to a point cloud with o3d
                # obb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(pcd))
                # obb = [obb.get_box_points()]
                # obb = np.asarray(obb)
                # obb = torch.from_numpy(obb).to('cuda')
                # # get the spatial similarity
                # spatial_sim = compute_iou_batch(obb, scene_gt_queries[scene_name]['gt_boxes'])

                # # get index of all values above a threshold
                # above_thresh = spatial_sim > 0.0
                # above_thresh = above_thresh.nonzero()[:,1]
                # # get the labels
                # gt_labels = [scene_gt_queries[scene_name]['gt_labels'][idx] for idx in above_thresh]
                # gt_sim = [spatial_sim[0, idx].item() for idx in above_thresh]
                # # get the class_nums
                # # class_nums = [semantic_classes.index(label) for label in gt_labels]

                # # create a point cloud from pcd
                # pcd_obj = o3d.geometry.PointCloud()
                # pcd_obj.points = o3d.utility.Vector3dVector(pcd)

                if GENERATE_EVAL_DATA:
                    # get the gt info
                    gt_pcd = scene_gt_queries[scene_name]['gt_pcd']
                    pcd_disp = scene_gt_queries[scene_name]['pcd_disp']
                    gt_annos = scene_gt_queries[scene_name]['gt_annos']

                    # match the pcd_obj to the gt_pcd
                    pred_gt_associations = compute_pred_gt_associations(torch.tensor(pcd).unsqueeze(0).cuda().contiguous().float(), torch.tensor(gt_pcd).unsqueeze(0).cuda().contiguous().float()) 
                    # print(pred_gt_associations[0].shape)
                    # get the annos
                    pred_annos = gt_annos[pred_gt_associations[0].detach().cpu().numpy()]
                    unique, counts = np.unique(pred_annos, return_counts=True)
                    pred_annos = [semantic_classes[anno] for anno in pred_annos if anno<len(semantic_classes)]
                    # get a count of each unique element in pred_annos
                    unique, counts = np.unique(pred_annos, return_counts=True)
                    # select the class with from nuique the maximum count
                    try:
                        gt_class = unique[np.argmax(counts)]
                        gt_labels = [gt_class]
                    except:
                        gt_class = []
                
                    # # get the ratio of the max value to the total
                    # ratios =[val/pred_gt_associations[0].shape[0] for val in counts]
                    # # get indices where ratio is above a threshold
                    # above_thresh = [i for i in range(len(ratios)) if ratios[i] > 0.3]
                    # # get the labels of the above threshold
                    # gt_labels = [unique[i] for i in above_thresh if unique[i] in target_classes]
                else:
                    gt_labels = [gt_class]

                # print(gt_labels)
                # print(object_tag)

                # # open the first element of image_names
                # for i in range(len(image_names)):
                #     # get the image name
                #     image_name = image_names[i].decode('utf-8')
                #     # load the image with PIL
                #     image = Image.open(image_name)
                #     # crop the image
                #     crop = crops[i]
                #     crop = crop.astype(int)
                #     image = image.crop(crop)
                #     # convert to numpy
                #     image = np.array(image)
                #     # display with matplotlib
                #     plt.figure(0)
                #     plt.imshow(image)
                #     plt.show()
                #     # close all figures
                #     plt.close('all')

                # # create obj_pcd
                # pcd_obj = o3d.geometry.PointCloud()
                # pcd_obj.points = o3d.utility.Vector3dVector(pcd)
                # # make the color red
                # pcd_obj.paint_uniform_color([1, 0, 0])
                # # display the point cloud
                # o3d.visualization.draw_geometries([pcd_disp, pcd_obj])


                # # get the maximum value and index
                # max_val, max_idx = spatial_sim.max(dim=1)
                # # get the label
                # gt_label = scene_gt_queries[scene_name]['gt_labels'][max_idx]
                # # get the class_num
                # cls_num = semantic_classes.index(gt_label)

                # print(gt_class)
                # print(gt_labels)
                # print(gt_sim)

                # if split == 'train':
                #     # only add the object if it is in the target classes
                #     if semantic_classes[cls_num] not in target_classes:
                #         continue

                # # get the mean visual features
                # xfI = np.mean(visual_features, axis=0)

                # append to scene objects
                scene_objects[scene_name]['object_feats'].append(visual_features)
                object_idx = len(scene_objects[scene_name]['object_feats']) - 1

                # get the class num
                scene_objects[scene_name]['object_cls'].append(gt_labels)

                # get the object file
                scene_objects[scene_name]['object_file'].append(instance_file)

                # # define if it is a top100 class or long tail class
                # flag = [False]
                # if semantic_classes[cls_num] not in train_classes:
                #     flag = [True]

                # # only add the object if it is in the target classes
                # if semantic_classes[cls_num] not in target_classes:
                #     continue

                # # get the text feature for the label
                # text_caption = [f'an image of a {gt_class}']
                # with torch.no_grad():
                #     caption_targets = clip_tokenizer(text_caption)
                #     caption_targets = caption_targets.to(device)
                #     text_features = CLIP_Net.encode_text(caption_targets)
                #     # convert to numpy
                #     text_features = text_features.unsqueeze(0).cpu().numpy()

                # if use_affordances:
                #     if semantic_classes[cls_num] in affordance_features.keys():
                #         aff_feat = affordance_features[semantic_classes[cls_num]].unsqueeze(0).cpu().numpy()
                #         # print(aff_feat.shape)
                #         # print(text_features.shape)
                #         text_features = np.concatenate((text_features, aff_feat), axis=0)

                #         captions = [f'an image of a {gt_class}', f'an image of {unambiguous_affordances[semantic_classes[cls_num]]}']

                #         flag = [False, True]

                # # iterate through each caption
                # for i in range(len(captions)):
                #     xfT = text_features[i]
                #     caption = captions[i]
                #     unseen = flag[i]

                #     # check if caption not in the data structure
                #     if caption not in scene_queries[scene_name]['captions']:
                #         scene_queries[scene_name]['captions'].append(caption)
                #         scene_queries[scene_name]['query_feats'].append(xfT)
                #         scene_queries[scene_name]['object_ids'].append([str(object_idx)])
                #         scene_queries[scene_name]['unseen'].append(unseen)
                #     else:
                #         scene_queries[scene_name]['object_ids'][scene_queries[scene_name]['captions'].index(caption)].append(str(object_idx))

        scene_queries = {}
        total_queries = 0
        missed_objects = 0
        # iterate through each scene and add the gt queries
        for scene_name in scene_gt_queries.keys():
            # print(scene_name)
            # print(scene_queries[scene_name]['captions'])
            scene_queries[scene_name] = {
                'captions': [],
                'query_feats': [],
                'object_ids': [],
                'unseen': []
            }
            for gt_query in set(scene_gt_queries[scene_name]['gt_labels']):
                total_queries+=1
                
                # # skip long-tail
                # if gt_query not in target_classes:
                #     print(gt_query)
                #     continue

                # generate feature
                text_caption = [f'an image of an {gt_query}']
                with torch.no_grad():
                    caption_targets = clip_tokenizer(text_caption)
                    caption_targets = caption_targets.to(device)
                    query_feat = CLIP_Net.encode_text(caption_targets)
                    # convert to numpy
                    query_feat = query_feat.cpu().numpy()

                # # check if gt_query is in the target classes
                # unseen = False
                # if not use_affordances:
                #     if gt_query not in train_classes:
                #         unseen = True
                unseen = True
                if gt_query in train_classes:
                    unseen = False

                # find the objects that match the query
                object_ids = []
                for object_idx, object_cls in enumerate(scene_objects[scene_name]['object_cls']):
                    if gt_query in object_cls:
                        object_ids.append(str(object_idx))
                # if object_ids is empty, then add a dummy object
                if len(object_ids) == 0:
                    object_ids = ['9999']
                    missed_objects+=1

                # append to the scene queries
                scene_queries[scene_name]['captions'].append(f'an image of a {gt_query}')
                scene_queries[scene_name]['query_feats'].append(query_feat)
                scene_queries[scene_name]['object_ids'].append(object_ids)
                scene_queries[scene_name]['unseen'].append(unseen)

                if use_affordances:
                    if gt_query in affordance_features.keys():
                        # get the affordance features
                        aff_feat = affordance_features[gt_query].cpu().numpy()
                        aff_caption = f'an image of a {unambiguous_affordances[gt_query]}'
                        # check if it is in the target classes
                        unseen = True
                        if unambiguous_affordances[gt_query] in train_classes:
                            unseen = False

                        # appemd the query
                        scene_queries[scene_name]['captions'].append(aff_caption)
                        scene_queries[scene_name]['query_feats'].append(aff_feat)
                        scene_queries[scene_name]['object_ids'].append(object_ids)
                        scene_queries[scene_name]['unseen'].append(unseen)
        
        # query_feats: text feature of unique class in the scene
        # captions: captions of unique class in the scene for explainability
        # object_feats: visual feature of all unique object instances
        # query_labels: 0 or 1 for each object instance

        # number of solvable queries 
        print(f'Mapped object accuracy: {(total_queries-missed_objects)/total_queries}')

        # get the maximum number of objects in a single scene, to pad the object features
        self.max_objects = max([len(scene_objects[scene]['object_feats']) for scene in scene_objects.keys()])

        self.query_feats = []
        self.captions = []
        self.object_feats = []
        self.query_labels = []
        self.unseen = []
        self.object_files = []
        self.n_objects = 0
        self.n_queries = 0
        for scene in scene_queries.keys():
            self.n_objects+=len(scene_objects[scene]['object_feats'])
            
            # object_feats = np.stack(scene_objects[scene]['object_feats'])
            # # pad with zeros to make object feats (max_objects, 512)
            # object_feats = np.pad(object_feats, ((0, self.max_objects-object_feats.shape[0]), (0,0)), 'constant', constant_values=0)

            # leave this as a list of tensors and save the padding for the data loader
            object_feats = scene_objects[scene]['object_feats']

            # prepare the object files
            object_files = scene_objects[scene]['object_file']
            # pad with None to make object files (max_objects)
            object_files = object_files + ['None']*(self.max_objects-len(object_files))

            for i in range(len(scene_queries[scene]['query_feats'])):
                # append the query features
                self.query_feats.append(scene_queries[scene]['query_feats'][i])
                self.captions.append(scene_queries[scene]['captions'][i])
                # self.query_labels.append(scene_queries[scene]['object_ids'][i])
                self.query_labels.append('_'.join(scene_queries[scene]['object_ids'][i]))
                # append the object features
                self.object_feats.append(object_feats)
                # append the object files
                self.object_files.append(object_files)
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
        object_feats = [np.mean(feats, axis=0) for feats in object_feats]
        # alternatively, we can iterate over the object feats and randomly select one features. This is useful for training image models
        # object_feats = [feats[np.random.randint(feats.shape[0])] for feats in object_feats]

        # stack the object_feats
        object_feats = np.stack(object_feats).squeeze(1)
        # pad with zeros to make object feats (max_objects, 512)
        object_feats = np.pad(object_feats, ((0, self.max_objects-object_feats.shape[0]), (0,0)), 'constant', constant_values=0)
        
        return self.query_feats[index], self.query_labels[index], self.captions[index], object_feats, self.unseen[index], self.object_files[index]
    
    def __len__(self):
        return len(self.captions)
    
    def get_gt_boxes(self, image_root, scene_name, target_classes):

        gt_seg_path = image_root+f'/data_download/complete_dataset/data/{scene_name}/scans/segments_anno.json'
        gt_pcd_path = image_root+f'/data_download/complete_dataset/data/{scene_name}/scans/mesh_aligned_0.05_semantic.ply'
        pcd_disp = o3d.io.read_point_cloud(gt_pcd_path)
        pcd = np.asarray(pcd_disp.points)

        # get the bounding box from the anno file
        gt_boxes = []
        gt_labels = []
        gt_annos = np.full(pcd.shape[0], len(target_classes), dtype=int)
        with open(gt_seg_path) as fin:
            anno = json.load(fin)
            for gt_instance in anno['segGroups']:
                # only add the object if it is in the target classes
                if gt_instance['label'] not in target_classes:
                    continue
                # get label and bounding box
                gt_labels.append(gt_instance['label'])
                point_indices = gt_instance['segments']
                points = pcd[point_indices]
                obb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(points))
                gt_boxes.append(obb.get_box_points())

                # add the points to gt_annos
                for idx in gt_instance['segments']:
                    gt_annos[idx] = target_classes.index(gt_instance['label'])

        gt_boxes = np.asarray(gt_boxes)
        # convert to torch
        gt_boxes = torch.from_numpy(gt_boxes).to('cuda')

        return gt_boxes, gt_labels, pcd, pcd_disp, gt_annos

    def get_gt_labels(self, image_root, scene_name, target_classes):

        gt_seg_path = image_root+f'/data_download/complete_dataset/data/{scene_name}/scans/segments_anno.json'
        segmentation_path = image_root+f'/data_download/complete_dataset/conceptgraph_data/{scene_name}/gsa_detections_none/'
        instance_path = image_root+f'/data_download/complete_dataset/data/{scene_name}/iphone/instance/'
        
        # # list all files in the segmentation path
        # seg_files = glob.glob(segmentation_path+'*.pkl.gz')
        # # observec instances
        # instances = []
        # # iterate through each file
        # for seg_file in seg_files:
        #     file_name = seg_file.split('/')[-1].replace('.pkl.gz', '.png')
        #     # open the instance file
        #     instance_file = instance_path+file_name
        #     # load the png with Image
        #     image = Image.open(instance_file)
        #     # convert to numpy
        #     image = np.array(image)
        #     # get the unique values
        #     unique = np.unique(image)
        #     # iterate through and calculate number of pixels
        #     for idx in unique:
        #         if idx == 0:
        #             continue
        #         num_pixels = np.sum(image==idx)
        #         if num_pixels > 400:
        #             instances.append(idx)
        # # count the number of times each unique instance appears in the list
        # instances = np.array(instances)
        # instances, counts = np.unique(instances, return_counts=True)
        # # convert to dict
        # instances = {instances[i]: counts[i] for i in range(len(instances))}

        # get the bounding box from the anno file
        gt_labels = []
        with open(gt_seg_path) as fin:
            anno = json.load(fin)
            for gt_instance in anno['segGroups']:
                # # skip if not in instances or if insufficient number
                # if gt_instance['id'] not in instances.keys():
                #     # print('instance not in instances')
                #     continue
                # if instances[gt_instance['id']] < 3:
                #     # print('instance not viewed')
                #     continue
                # only add the object if it is in the target classes
                if gt_instance['label'] not in target_classes:
                    continue
                # get label and bounding box
                gt_labels.append(gt_instance['label'])

        return gt_labels

class ConceptgraphCaptionPseudoLabel(Dataset):
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
                 split='val', pseudo_thresh=0.3, pseudo_method='scene', use_affordances=False,
                 ):

        print(pseudo_method, pseudo_thresh)
        
        # self.image_root = os.path.expanduser(image_root)
        # self.caption_root = os.path.expanduser(caption_root)
        # device = 'cuda'
        # CLIP_Net = load_model(device='cuda', model_path=None)
        # CLIP_Net.eval()
    
        # Initialize the CLIP model used in conceptgraphs
        device = 'cuda'
        CLIP_Net, _, clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", "laion2b_s32b_b79k"
        )
        CLIP_Net = CLIP_Net.to(device)
        CLIP_Net.eval()
        clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

        # if isinstance(target_classes, str):
        #     with open(target_classes) as fin:
        #         _classes = [int(line.strip().split('_')[1]) - 1 for line in fin]
        #     target_classes = _classes
        
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

        # # get the semantic classes
        # semantic_classes = image_root+f'/data_download/complete_dataset/metadata/semantic_classes.txt'
        # top100_classes = image_root+f'/data_download/complete_dataset/metadata/semantic_benchmark/top100.txt'
        # with open(semantic_classes) as fin:
        #     semantic_classes = [line.strip() for line in fin]
        # # print(semantic_classes)

        # with open(top100_classes) as fin:
        #     top100_classes = [line.strip() for line in fin]
        # # print(top100_classes)

        # # define the training and testing classses
        # if seen_classes == 'top100':
        #     train_classes = top100_classes
        #     test_classes = semantic_classes
        #     calib_classes = top100_classes
        # elif seen_classes == 'half':
        #     train_classes = semantic_classes[0::2]
        #     test_classes = semantic_classes
        #     calib_classes = semantic_classes[0::2]
        # elif seen_classes == 'top100_half':
        #     train_classes = top100_classes[0:53]
        #     test_classes = semantic_classes
        #     calib_classes = top100_classes[0:53]

        # # define the target classes
        # if split == 'train':
        #     # target_classes = top100_classes
        #     target_classes = train_classes
        #     target_scenes = [] # scenes[0:50]
        # elif split == 'calib':
        #     target_classes = calib_classes
        #     target_scenes = [] #scenes[0:50]
        #     # scenes = scenes[-50:]
        # else:
        #     target_classes = test_classes
        #     target_scenes = scenes
        
        # # remove some classes
        # exclude_classes = ['ceiling', 'floor', 'wall']
        # target_classes = [cls for cls in target_classes if cls not in exclude_classes]

        # self.target_classes = target_classes

        ###################### PREPARE THE DATASET FOR TRAINING #####################

        # # pseudo-label queries for the  unlabelled scenes
        # with torch.no_grad():
        #     # either do all classes, or just the unseen classes
        #     # unseen_classes = [cls for cls in semantic_classes if cls not in target_classes]
        #     unseen_classes = [cls for cls in semantic_classes]
        #     # remove exclude classes
        #     unseen_classes = [cls for cls in unseen_classes if cls not in exclude_classes]
        #     unseen_queries = [f'an image of a {cls}' for cls in unseen_classes]
        #     unseen_targets = target_transform(unseen_queries)
        #     unseen_targets = unseen_targets.to(device)
        #     unseen_feats = CLIP_Net.encode_text(unseen_targets)

        #     # get the "other" class embedding
        #     other_target = target_transform(['an image of an object'])
        #     other_target = other_target.to(device)
        #     other_feat = CLIP_Net.encode_text(other_target)
                            
        # This can be editted dynamically as we change the training approach etc..
        scene_objects = {}
        scene_queries = {}
        pseudo_objects = {}
        # list all files in the caption root
        instance_files = glob.glob(caption_root+f'/{dataset_split}/*.h5')
        # iterate through each file
        for instance_file in tqdm.tqdm(instance_files):
            scene_name = instance_file.split('/')[-1].split('_')[0]

            # if scene_name not in scenes:
            #     continue

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
            
            # # attempt to open the caption data
            # with h5py.File(caption_file, 'r') as f: 
            #     # print(f['captions'][()].decode('utf-8'))
            #     try:
            #         object_tag = f['object_tag'][()].decode('utf-8').replace('_', ' ')
            #         object_description = f['object_description'][()].decode('utf-8').replace('_', ' ')
            #         object_affordance = f['object_affordance'][()].decode('utf-8').replace('_', ' ')
            #         # ignore invalid
            #         if object_tag == 'invalid':
            #             continue
            #         # print(object_tag, object_description, object_affordance)
            #     except KeyError:
            #         continue
            
            # open the object info file
            with h5py.File(instance_file, 'r') as f:
                try:
                    # get the object feature data
                    image_names = f['image_names']
                    crops = f['crops']
                    visual_features = f['visual_features'][()]
                    xfI = f['object_feat'][()]

                    # get the caption data
                    init_captions = f['captions']
                    object_tag = f['object_tags'][()].decode('utf-8')
                    object_description = f['object_description'][()].decode('utf-8')
                    object_affordance = f['object_affordance'][()].decode('utf-8')
                except Exception as e:
                    # print(e)
                    continue

                # cls_num = class_nums[0]
                # if split == 'train':
                # if split == 'train' or split == 'calib':
                
                # get the mean visual features
                # xfI = np.mean(visual_features, axis=0)

                # # if a labelled scene, we skip all objects that are not in the target classes
                # if scene_name in target_scenes:
                #     # ignore if not in target classes
                #     if semantic_classes[cls_num] not in target_classes:
                #         continue
                #     # append object
                #     else:
                #         scene_objects[scene_name]['object_feats'].append(xfI)

                #         object_idx = len(scene_objects[scene_name]['object_feats']) - 1
                        
                #         # get the class num
                #         scene_objects[scene_name]['class_nums'].append(cls_num)

                #         # define if it is a top100 class or long tail class
                #         unseen = False
                #         if semantic_classes[cls_num] not in train_classes:
                #             unseen = True

                #         # only add the object if it is in the target classes
                #         # if semantic_classes[cls_num] not in target_classes:
                #         #     continue

                #         # iterate through each caption
                #         for i in range(len(captions)):
                #             xfT = text_features[i]
                #             caption = captions[i]

                #             # check if caption not in the data structure
                #             if caption not in scene_queries[scene_name]['captions']:
                #                 scene_queries[scene_name]['captions'].append(caption)
                #                 scene_queries[scene_name]['query_feats'].append(xfT)
                #                 scene_queries[scene_name]['object_ids'].append([str(object_idx)])
                #                 scene_queries[scene_name]['unseen'].append(unseen)
                #             else:
                #                 scene_queries[scene_name]['object_ids'][scene_queries[scene_name]['captions'].index(caption)].append(str(object_idx))
                                
            # otherwise, we attempt to pseudo-label the object
            # else:  
            captions = [object_tag, object_description, object_affordance]
            # pseudo-label queries for the  unlabelled scenes
            with torch.no_grad():
                caption_query = [f'an image of a {captions[0]}', f'an image of a {captions[1]}', f'an image of an object {captions[2]}']
                caption_targets = clip_tokenizer(caption_query)
                caption_targets = caption_targets.to(device)
                caption_feat = CLIP_Net.encode_text(caption_targets)

            # get similarity between the object features and caption feature
            img_embedding = torch.tensor(xfI).to(device)
            caption_embedding = torch.tensor(caption_feat).to(device)
            img_embedding = img_embedding / img_embedding.norm(dim=-1, keepdim=True)
            caption_embedding = caption_embedding / caption_embedding.norm(dim=-1, keepdim=True)
            logits = torch.matmul(caption_embedding, img_embedding.T).squeeze(0)
            # print(captions[0], semantic_classes[cls_num], logits)

            # filter poor quality pseudo-labels
            if logits[0] < pseudo_thresh:
                continue

            # # only select the visual features that are above the threshold
            # norm_visual_feat = [torch.tensor(visual_features[i]).to(device) for i in range(visual_features.shape[0])]
            # norm_visual_feat = [feat / feat.norm(dim=-1, keepdim=True) for feat in norm_visual_feat]
            # # stack the features
            # norm_visual_feat = torch.stack(norm_visual_feat).squeeze(1)
            # # get the similarity
            # logits = torch.matmul(caption_embedding, norm_visual_feat.T).squeeze(0)
            # # only keep the features above the threshold
            # above_thresh = logits > pseudo_thresh
            # # convert to numpy
            # above_thresh = above_thresh.cpu().numpy()[0]
            # # if no features above threshold, continue
            # if above_thresh.sum() == 0:
            #     continue
            # # select the visual features
            # visual_features = visual_features[above_thresh]

            # # only select the visual features that are above the threshold
            # norm_visual_feat = [torch.tensor(visual_features[i]).to(device) for i in range(visual_features.shape[0])]
            # norm_visual_feat = [feat / feat.norm(dim=-1, keepdim=True) for feat in norm_visual_feat]
            # # stack the features
            # norm_visual_feat = torch.stack(norm_visual_feat).squeeze(1)
            # # get the similarity
            # logits = torch.matmul(caption_embedding, norm_visual_feat.T).squeeze(0)
            # # find the index of the maximum value
            # max_val, max_idx = logits.max(dim=0)
            # # skip if max value is below threshold
            # if max_val[0] < pseudo_thresh:
            #     continue
            # # select the visual feature at this index
            # visual_features = visual_features[max_idx[0]]
            # visual_features = np.expand_dims(visual_features,0)

            scene_objects[scene_name]['object_feats'].append(visual_features)

            object_idx = len(scene_objects[scene_name]['object_feats']) - 1
            
            # get the class num
            # scene_objects[scene_name]['class_nums'].append(cls_num)

            # define if it is a top100 class or long tail class
            flag = [True]
            # if semantic_classes[cls_num] not in train_classes:
            #     flag = [True]

            # generate the captions to be used as pseudo-labels
            if use_affordances:
                captions = [caption_query[0], caption_query[2]]
                text_features = [caption_feat[0].unsqueeze(0).cpu().numpy(), caption_feat[2].unsqueeze(0).cpu().numpy()]
                flag = [False, True]
            else:
                captions = [caption_query[0]]
                text_features = [caption_feat[0].unsqueeze(0).cpu().numpy()]

            # print(captions)
            # iterate through each caption
            for i in range(len(captions)): 
                caption = captions[i]  
                xfT = text_features[i] 
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
            
            # object_feats = np.stack(scene_objects[scene]['object_feats'])
            # object_feats = object_feats.squeeze(1)
            # pad with zeros to make object feats (max_objects, n)
            # object_feats = np.pad(object_feats, ((0, self.max_objects-object_feats.shape[0]), (0,0)), 'constant', constant_values=0)
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
        object_feats = np.stack(object_feats).squeeze(1)
        # pad with zeros to make object feats (max_objects, 512)
        object_feats = np.pad(object_feats, ((0, self.max_objects-object_feats.shape[0]), (0,0)), 'constant', constant_values=0)
        
        return self.query_feats[index], self.query_labels[index], self.captions[index], object_feats, self.unseen[index]
    
    def __len__(self):
        return len(self.captions)

class ConceptgraphCosinePseudo(Dataset):
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
                 target_classes, split='val', thresh=0.3,
                 transform=None, target_transform=None, seen_classes='top100', use_affordances = False
                 ):

        # self.image_root = os.path.expanduser(image_root)
        # self.caption_root = os.path.expanduser(caption_root)
        # device = 'cuda'
        # CLIP_Net = load_model(device='cuda', model_path=None)
        # CLIP_Net.eval()

        device = 'cuda'
        CLIP_Net, _, clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", "laion2b_s32b_b79k"
        )
        CLIP_Net = CLIP_Net.to(device)
        CLIP_Net.eval()
        clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

        # if isinstance(target_classes, str):
        #     with open(target_classes) as fin:
        #         _classes = [int(line.strip().split('_')[1]) - 1 for line in fin]
        #     target_classes = _classes
        

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
        # print(semantic_classes)

        with open(top100_classes) as fin:
            top100_classes = [line.strip() for line in fin]
        # print(top100_classes)

        # generate text features
        text_caption = [f'an image of a {query}' for query in semantic_classes]
        with torch.no_grad():
            caption_targets = clip_tokenizer(text_caption)
            caption_targets = caption_targets.to(device)
            text_feats = CLIP_Net.encode_text(caption_targets)

        # open the unambiguous affordances as a json file
        with open(image_root+f'/data_download/complete_dataset/metadata/unambiguous_affordances.json') as fin:
            unambiguous_affordances = json.load(fin)

        affordance_features = {}
        for affordance, object_top100 in unambiguous_affordances.items():
            with torch.no_grad():
                caption_query = [f'an image of {affordance}']
                caption_targets = clip_tokenizer(caption_query)
                caption_targets = caption_targets.to(device)
                caption_feat = CLIP_Net.encode_text(caption_targets)
                affordance_features[object_top100] = caption_feat
        unambiguous_affordances = {v: k for k, v in unambiguous_affordances.items()}

        # define the training and testing classses
        if seen_classes == 'top100':
            train_classes = top100_classes
            test_classes = semantic_classes
            calib_classes = top100_classes
        elif seen_classes == 'half':
            train_classes = semantic_classes[0::2]
            test_classes = semantic_classes
            calib_classes = semantic_classes[0::2]
        elif seen_classes == 'top100_half':
            train_classes = top100_classes[0:53]
            test_classes = semantic_classes
            calib_classes = top100_classes[0:53]

        # # define the target classes
        # if split == 'train':
        #     # target_classes = top100_classes
        #     target_classes = train_classes
        # elif split == 'calib':
        #     target_classes = calib_classes
        #     # scenes = scenes[-50:]
        # else:
        #     target_classes = test_classes
        target_classes = test_classes
        
        # remove some classes
        exclude_classes = ['ceiling', 'floor', 'wall', 'object']
        target_classes = [cls for cls in target_classes if cls not in exclude_classes]
        self.target_classes = target_classes

        ###################### PREPARE THE DATASET FOR TRAINING #####################

        # This can be editted dynamically as we change the training approach etc..

        scene_objects = {}
        scene_gt_queries = {}
        # list all files in the caption root
        instance_files = glob.glob(caption_root+f'/{dataset_split}/*.h5')
        # iterate through each file
        for instance_file in tqdm.tqdm(instance_files):
            scene_name = instance_file.split('/')[-1].split('_')[0]

            # if scene_name not in target_scenes:
            #     continue

            # if scene_name not in ['56a0ec536c', '8b5caf3398', '41b00feddb', '98b4ec142f', '7b6477cb95']:
            #     continue

            with open(image_root+f'/data_download/complete_dataset/data/{scene_name}/scans/segments_anno.json') as fin:
                segments = json.load(fin)

            # initialise the data structures for the scene
            if scene_name not in scene_objects.keys():
                # dictionary to store the object features
                scene_objects[scene_name] = {
                    'object_feats': [],
                    'object_cls': [],
                    'object_file': []
                }

                # to store info about the gt objects
                scene_gt_queries[scene_name] = {}

                # # open the segments metadata
                # with open(image_root+f'/data_download/complete_dataset/data/{scene_name}/scans/segments_anno.json') as fin:
                #     segments = json.load(fin)
                #     scene_gt_queries[scene_name] = set()
                #     for gt_obj in segments['segGroups']:
                #         if gt_obj['label'] in self.target_classes:
                #             scene_gt_queries[scene_name].add(gt_obj['label'])

                # get the labels of objects that are visible in the scene
                gt_labels = self.get_gt_labels(image_root, scene_name, target_classes)
                scene_gt_queries[scene_name]['gt_labels'] = gt_labels
            
            # open the object info file
            with h5py.File(instance_file, 'r') as f:
                try:
                    # get the object feature data
                    image_names = f['image_names']
                    crops = f['crops']
                    visual_features = f['visual_features'][()]
                    xfI = f['object_feat'][()]
                    gt_class = f['gt_class'][()].decode('utf-8')
                    flag = [False]
                    captions = [f'an image of a {gt_class}']
                    pcd = f['pcd'][()]

                    # get the caption data
                    init_captions = f['captions']
                    object_tag = f['object_tags'][()].decode('utf-8')
                    object_description = f['object_description'][()].decode('utf-8')
                    object_affordance = f['object_affordance'][()].decode('utf-8')
                except Exception as e:
                    # print(e)
                    continue

                # get the clip feature for each class in the scene
                scene_queries = scene_gt_queries[scene_name]['gt_labels']

                # get the indeces of the scene queries in semantic classes
                scene_ids = [semantic_classes.index(cls) for cls in scene_queries]
                # extract the query_feats
                query_feats = text_feats[scene_ids]
                    
                # get similarity between the object features and caption feature
                img_embedding = torch.tensor(xfI).to(device)
                caption_embedding = torch.tensor(query_feats).to(device)
                img_embedding = img_embedding / img_embedding.norm(dim=-1, keepdim=True)
                caption_embedding = caption_embedding / caption_embedding.norm(dim=-1, keepdim=True)
                logits = torch.matmul(caption_embedding, img_embedding.T).squeeze(0)

                # get the maximum value of the logits
                max_val, max_idx = logits.max(dim=0)
                # if less than thresh, ignore
                if max_val < thresh:
                    continue
                # get the label of the maximum class
                max_class = scene_queries[max_idx]
                # print(max_class, gt_class)

                # gt_labels is a list of all correct object classes
                gt_labels = [max_class]

                # # get the mean visual features
                # xfI = np.mean(visual_features, axis=0)

                # append to scene objects
                scene_objects[scene_name]['object_feats'].append(visual_features)
                object_idx = len(scene_objects[scene_name]['object_feats']) - 1

                # get the class num
                scene_objects[scene_name]['object_cls'].append(gt_labels)

                # get the object file
                scene_objects[scene_name]['object_file'].append(instance_file)

        scene_queries = {}
        total_queries = 0
        missed_objects = 0
        # iterate through each scene and add the gt queries
        for scene_name in scene_gt_queries.keys():
            # print(scene_name)
            # print(scene_queries[scene_name]['captions'])
            scene_queries[scene_name] = {
                'captions': [],
                'query_feats': [],
                'object_ids': [],
                'unseen': []
            }
            for gt_query in set(scene_gt_queries[scene_name]['gt_labels']):
                total_queries+=1
                
                # generate feature
                text_caption = [f'an image of an {gt_query}']
                with torch.no_grad():
                    caption_targets = clip_tokenizer(text_caption)
                    caption_targets = caption_targets.to(device)
                    query_feat = CLIP_Net.encode_text(caption_targets)
                    # convert to numpy
                    query_feat = query_feat.cpu().numpy()

                # check if gt_query is in the target classes
                unseen = False
                if not use_affordances:
                    if gt_query not in train_classes:
                        unseen = True

                # find the objects that match the query
                object_ids = []
                for object_idx, object_cls in enumerate(scene_objects[scene_name]['object_cls']):
                    if gt_query in object_cls:
                        object_ids.append(str(object_idx))
                # if object_ids is empty, then add a dummy object
                if len(object_ids) == 0:
                    object_ids = ['9999']
                    missed_objects+=1

                # append to the scene queries
                scene_queries[scene_name]['captions'].append(f'an image of a {gt_query}')
                scene_queries[scene_name]['query_feats'].append(query_feat)
                scene_queries[scene_name]['object_ids'].append(object_ids)
                scene_queries[scene_name]['unseen'].append(unseen)

                if use_affordances:
                    if gt_query in affordance_features.keys():
                        # get the affordance features
                        aff_feat = affordance_features[gt_query].cpu().numpy()
                        aff_caption = f'an image of {unambiguous_affordances[gt_query]}'
                        unseen = True
                        
                        # appemd the query
                        scene_queries[scene_name]['captions'].append(aff_caption)
                        scene_queries[scene_name]['query_feats'].append(aff_feat)
                        scene_queries[scene_name]['object_ids'].append(object_ids)
                        scene_queries[scene_name]['unseen'].append(unseen)
        
        # query_feats: text feature of unique class in the scene
        # captions: captions of unique class in the scene for explainability
        # object_feats: visual feature of all unique object instances
        # query_labels: 0 or 1 for each object instance

        # number of solvable queries 
        print(f'Mapped object accuracy: {(total_queries-missed_objects)/total_queries}')

        # get the maximum number of objects in a single scene, to pad the object features
        self.max_objects = max([len(scene_objects[scene]['object_feats']) for scene in scene_objects.keys()])

        self.query_feats = []
        self.captions = []
        self.object_feats = []
        self.query_labels = []
        self.unseen = []
        self.object_files = []
        self.n_objects = 0
        self.n_queries = 0
        for scene in scene_queries.keys():
            self.n_objects+=len(scene_objects[scene]['object_feats'])
            
            # object_feats = np.stack(scene_objects[scene]['object_feats'])
            # # pad with zeros to make object feats (max_objects, 512)
            # object_feats = np.pad(object_feats, ((0, self.max_objects-object_feats.shape[0]), (0,0)), 'constant', constant_values=0)

            # leave this as a list of tensors and save the padding for the data loader
            object_feats = scene_objects[scene]['object_feats']

            # prepare the object files
            object_files = scene_objects[scene]['object_file']
            # pad with None to make object files (max_objects)
            object_files = object_files + ['None']*(self.max_objects-len(object_files))

            for i in range(len(scene_queries[scene]['query_feats'])):
                # append the query features
                self.query_feats.append(scene_queries[scene]['query_feats'][i])
                self.captions.append(scene_queries[scene]['captions'][i])
                # self.query_labels.append(scene_queries[scene]['object_ids'][i])
                self.query_labels.append('_'.join(scene_queries[scene]['object_ids'][i]))
                # append the object features
                self.object_feats.append(object_feats)
                # append the object files
                self.object_files.append(object_files)
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
        object_feats = [np.mean(feats, axis=0) for feats in object_feats]
        # alternatively, we can iterate over the object feats and randomly select one features. This is useful for training image models
        # object_feats = [feats[np.random.randint(feats.shape[0])] for feats in object_feats]

        # stack the object_feats
        object_feats = np.stack(object_feats).squeeze(1)
        # pad with zeros to make object feats (max_objects, 512)
        object_feats = np.pad(object_feats, ((0, self.max_objects-object_feats.shape[0]), (0,0)), 'constant', constant_values=0)
        
        return self.query_feats[index], self.query_labels[index], self.captions[index], object_feats, self.unseen[index], self.object_files[index]
    
    def __len__(self):
        return len(self.captions)
    
    def get_gt_boxes(self, image_root, scene_name, target_classes):

        gt_seg_path = image_root+f'/data_download/complete_dataset/data/{scene_name}/scans/segments_anno.json'
        gt_pcd_path = image_root+f'/data_download/complete_dataset/data/{scene_name}/scans/mesh_aligned_0.05_semantic.ply'
        pcd_disp = o3d.io.read_point_cloud(gt_pcd_path)
        pcd = np.asarray(pcd_disp.points)

        # get the bounding box from the anno file
        gt_boxes = []
        gt_labels = []
        gt_annos = np.full(pcd.shape[0], len(target_classes), dtype=int)
        with open(gt_seg_path) as fin:
            anno = json.load(fin)
            for gt_instance in anno['segGroups']:
                # only add the object if it is in the target classes
                if gt_instance['label'] not in target_classes:
                    continue
                # get label and bounding box
                gt_labels.append(gt_instance['label'])
                point_indices = gt_instance['segments']
                points = pcd[point_indices]
                obb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(points))
                gt_boxes.append(obb.get_box_points())

                # add the points to gt_annos
                for idx in gt_instance['segments']:
                    gt_annos[idx] = target_classes.index(gt_instance['label'])

        gt_boxes = np.asarray(gt_boxes)
        # convert to torch
        gt_boxes = torch.from_numpy(gt_boxes).to('cuda')

        return gt_boxes, gt_labels, pcd, pcd_disp, gt_annos

    def get_gt_labels(self, image_root, scene_name, target_classes):

        gt_seg_path = image_root+f'/data_download/complete_dataset/data/{scene_name}/scans/segments_anno.json'
        segmentation_path = image_root+f'/data_download/complete_dataset/conceptgraph_data/{scene_name}/gsa_detections_none/'
        instance_path = image_root+f'/data_download/complete_dataset/data/{scene_name}/iphone/instance/'
        
        # # list all files in the segmentation path
        # seg_files = glob.glob(segmentation_path+'*.pkl.gz')
        # # observec instances
        # instances = []
        # # iterate through each file
        # for seg_file in seg_files:
        #     file_name = seg_file.split('/')[-1].replace('.pkl.gz', '.png')
        #     # open the instance file
        #     instance_file = instance_path+file_name
        #     # load the png with Image
        #     image = Image.open(instance_file)
        #     # convert to numpy
        #     image = np.array(image)
        #     # get the unique values
        #     unique = np.unique(image)
        #     # iterate through and calculate number of pixels
        #     for idx in unique:
        #         if idx == 0:
        #             continue
        #         num_pixels = np.sum(image==idx)
        #         if num_pixels > 400:
        #             instances.append(idx)
        # # count the number of times each unique instance appears in the list
        # instances = np.array(instances)
        # instances, counts = np.unique(instances, return_counts=True)
        # # convert to dict
        # instances = {instances[i]: counts[i] for i in range(len(instances))}

        # get the bounding box from the anno file
        gt_labels = []
        with open(gt_seg_path) as fin:
            anno = json.load(fin)
            for gt_instance in anno['segGroups']:
                # # skip if not in instances or if insufficient number
                # if gt_instance['id'] not in instances.keys():
                #     # print('instance not in instances')
                #     continue
                # if instances[gt_instance['id']] < 3:
                #     # print('instance not viewed')
                #     continue
                # only add the object if it is in the target classes
                if gt_instance['label'] not in target_classes:
                    continue
                # get label and bounding box
                gt_labels.append(gt_instance['label'])

        return gt_labels
    
class ConceptgraphContinualPseudoLabel(Dataset):
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
                 split='val', pseudo_thresh=0.3, pseudo_method='scene', use_affordances=False,
                 ):

        print(pseudo_method, pseudo_thresh)
        
        # self.image_root = os.path.expanduser(image_root)
        # self.caption_root = os.path.expanduser(caption_root)
        # device = 'cuda'
        # CLIP_Net = load_model(device='cuda', model_path=None)
        # CLIP_Net.eval()
    
        # Initialize the CLIP model used in conceptgraphs
        device = 'cuda'
        CLIP_Net, _, clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", "laion2b_s32b_b79k"
        )
        CLIP_Net = CLIP_Net.to(device)
        CLIP_Net.eval()
        clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

        # if isinstance(target_classes, str):
        #     with open(target_classes) as fin:
        #         _classes = [int(line.strip().split('_')[1]) - 1 for line in fin]
        #     target_classes = _classes
        
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
        exclude_classes = ['ceiling', 'floor', 'wall', 'object']
        target_classes = [cls for cls in semantic_classes if cls not in exclude_classes]
        top50  = [c for c in top100_classes[0:53] if c not in exclude_classes]

        # open the unambiguous affordances as a json file
        with open(image_root+f'/data_download/complete_dataset/metadata/unambiguous_affordances.json') as fin:
            unambiguous_affordances = json.load(fin)

        # affordance_features = {}
        # for affordance, object_top100 in unambiguous_affordances.items():
        #     with torch.no_grad():
        #         caption_query = [f'an image of {affordance}']
        #         caption_targets = clip_tokenizer(caption_query)
        #         caption_targets = caption_targets.to(device)
        #         caption_feat = CLIP_Net.encode_text(caption_targets)
        #         affordance_features[object_top100] = caption_feat
        unambiguous_affordances = {v: k for k, v in unambiguous_affordances.items()}
        affordance_list = list(unambiguous_affordances.values())
        # sort the list for consistency
        affordance_list.sort()

        # generate a list of core concepts or train_classes
        # 10 random indices from the top100 classes
        top100_idx = [23, 37, 25, 41, 22, 7, 48, 46, 9, 3]
        longtail_idx = [1233, 1586, 1180, 1047, 1020, 585, 231, 343, 101, 843, 1332, 1050, 1022, 1331, 1188, 228, 1181, 393, 1416, 201, 1178, 525, 709, 745, 798, 496, 1356, 1188, 368, 367, 1313, 885, 1284, 1134, 1597, 496, 1278, 153, 980, 1304, 1346, 850, 969, 978, 1542, 879, 1050, 102, 134, 1273, 736, 629, 1592, 1461, 625, 1462, 1413, 1063, 1105, 978, 838, 853, 1268, 248, 1262, 373, 885, 1368, 757, 747, 1284, 601, 570, 827, 1048, 147, 296, 1512, 873, 1011, 1264, 1224, 1048, 945, 1, 654, 393, 1557, 216, 495, 1110, 3, 562, 721, 1445, 1545, 350, 120, 27, 1447]
        affordance_idx = [12, 3, 32, 37, 8, 11, 14, 18, 2, 31]
        ignore_top_50 = [cls for cls in target_classes if cls not in top50]

        # define the training classes
        train_classes = [top100_classes[idx] for idx in top100_idx]
        # add the longtail classes
        train_classes += [ignore_top_50[idx] for idx in longtail_idx]
        # add values from the affordances
        train_classes += [affordance_list[idx] for idx in affordance_idx]
        print(train_classes)

        # encode the train classes
        with torch.no_grad():
            caption_query = [f'an image of a {concept}' for concept in train_classes]
            caption_targets = clip_tokenizer(caption_query)
            caption_targets = caption_targets.to(device)
            core_feats = CLIP_Net.encode_text(caption_targets)

        # # define the training and testing classses
        # if seen_classes == 'top100':
        #     train_classes = top100_classes
        #     test_classes = semantic_classes
        #     calib_classes = top100_classes
        # elif seen_classes == 'half':
        #     train_classes = semantic_classes[0::2]
        #     test_classes = semantic_classes
        #     calib_classes = semantic_classes[0::2]
        # elif seen_classes == 'top100_half':
        #     train_classes = top100_classes[0:53]
        #     test_classes = semantic_classes
        #     calib_classes = top100_classes[0:53]

        # # define the target classes
        # if split == 'train':
        #     # target_classes = top100_classes
        #     target_classes = train_classes
        #     target_scenes = [] # scenes[0:50]
        # elif split == 'calib':
        #     target_classes = calib_classes
        #     target_scenes = [] #scenes[0:50]
        #     # scenes = scenes[-50:]
        # else:
        #     target_classes = test_classes
        #     target_scenes = scenes
        
        # # remove some classes
        # exclude_classes = ['ceiling', 'floor', 'wall']
        # target_classes = [cls for cls in target_classes if cls not in exclude_classes]

        # self.target_classes = target_classes

        ###################### PREPARE THE DATASET FOR TRAINING #####################

        # # pseudo-label queries for the  unlabelled scenes
        # with torch.no_grad():
        #     # either do all classes, or just the unseen classes
        #     # unseen_classes = [cls for cls in semantic_classes if cls not in target_classes]
        #     unseen_classes = [cls for cls in semantic_classes]
        #     # remove exclude classes
        #     unseen_classes = [cls for cls in unseen_classes if cls not in exclude_classes]
        #     unseen_queries = [f'an image of a {cls}' for cls in unseen_classes]
        #     unseen_targets = target_transform(unseen_queries)
        #     unseen_targets = unseen_targets.to(device)
        #     unseen_feats = CLIP_Net.encode_text(unseen_targets)

        #     # get the "other" class embedding
        #     other_target = target_transform(['an image of an object'])
        #     other_target = other_target.to(device)
        #     other_feat = CLIP_Net.encode_text(other_target)
                            
        # This can be editted dynamically as we change the training approach etc..
        scene_objects = {}
        scene_queries = {}
        pseudo_objects = {}
        # list all files in the caption root
        instance_files = glob.glob(caption_root+f'/{dataset_split}/*.h5')
        # iterate through each file
        for instance_file in tqdm.tqdm(instance_files):
            scene_name = instance_file.split('/')[-1].split('_')[0]

            # if scene_name not in scenes:
            #     continue

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
            
            # # attempt to open the caption data
            # with h5py.File(caption_file, 'r') as f: 
            #     # print(f['captions'][()].decode('utf-8'))
            #     try:
            #         object_tag = f['object_tag'][()].decode('utf-8').replace('_', ' ')
            #         object_description = f['object_description'][()].decode('utf-8').replace('_', ' ')
            #         object_affordance = f['object_affordance'][()].decode('utf-8').replace('_', ' ')
            #         # ignore invalid
            #         if object_tag == 'invalid':
            #             continue
            #         # print(object_tag, object_description, object_affordance)
            #     except KeyError:
            #         continue
            
            # open the object info file
            with h5py.File(instance_file, 'r') as f:
                try:
                    # get the object feature data
                    image_names = f['image_names']
                    crops = f['crops']
                    visual_features = f['visual_features'][()]
                    xfI = f['object_feat'][()]

                    # get the caption data
                    init_captions = f['captions']
                    object_tag = f['object_tags'][()].decode('utf-8')
                    object_description = f['object_description'][()].decode('utf-8')
                    object_affordance = f['object_affordance'][()].decode('utf-8')
                except Exception as e:
                    # print(e)
                    continue

                # cls_num = class_nums[0]
                # if split == 'train':
                # if split == 'train' or split == 'calib':
                
                # get the mean visual features
                # xfI = np.mean(visual_features, axis=0)

                # # if a labelled scene, we skip all objects that are not in the target classes
                # if scene_name in target_scenes:
                #     # ignore if not in target classes
                #     if semantic_classes[cls_num] not in target_classes:
                #         continue
                #     # append object
                #     else:
                #         scene_objects[scene_name]['object_feats'].append(xfI)

                #         object_idx = len(scene_objects[scene_name]['object_feats']) - 1
                        
                #         # get the class num
                #         scene_objects[scene_name]['class_nums'].append(cls_num)

                #         # define if it is a top100 class or long tail class
                #         unseen = False
                #         if semantic_classes[cls_num] not in train_classes:
                #             unseen = True

                #         # only add the object if it is in the target classes
                #         # if semantic_classes[cls_num] not in target_classes:
                #         #     continue

                #         # iterate through each caption
                #         for i in range(len(captions)):
                #             xfT = text_features[i]
                #             caption = captions[i]

                #             # check if caption not in the data structure
                #             if caption not in scene_queries[scene_name]['captions']:
                #                 scene_queries[scene_name]['captions'].append(caption)
                #                 scene_queries[scene_name]['query_feats'].append(xfT)
                #                 scene_queries[scene_name]['object_ids'].append([str(object_idx)])
                #                 scene_queries[scene_name]['unseen'].append(unseen)
                #             else:
                #                 scene_queries[scene_name]['object_ids'][scene_queries[scene_name]['captions'].index(caption)].append(str(object_idx))
                                
            # otherwise, we attempt to pseudo-label the object
            # else:  
            captions = [object_tag, object_description, object_affordance]
            # pseudo-label queries for the  unlabelled scenes
            with torch.no_grad():
                caption_query = [f'an image of a {captions[0]}', f'an image of a {captions[1]}', f'an image of an object {captions[2]}']
                caption_targets = clip_tokenizer(caption_query)
                caption_targets = caption_targets.to(device)
                caption_feat = CLIP_Net.encode_text(caption_targets)

            # get similarity between the object features and caption feature
            img_embedding = torch.tensor(xfI).to(device)
            caption_embedding = torch.tensor(caption_feat).to(device)
            img_embedding = img_embedding / img_embedding.norm(dim=-1, keepdim=True)
            caption_embedding = caption_embedding / caption_embedding.norm(dim=-1, keepdim=True)
            logits = torch.matmul(caption_embedding, img_embedding.T).squeeze(0)
            caption_sim = logits[0]
            # print(captions[0], semantic_classes[cls_num], logits)

            # get the sim between the object and the core concepts
            core_embedding = core_feats / core_feats.norm(dim=-1, keepdim=True)
            logits = torch.matmul(core_embedding, img_embedding.T).squeeze(0)
            # get the index values of all values greater than 0.3
            above_thresh = logits > 0.3
            above_thresh = [train_classes[i] for i in range(len(above_thresh)) if above_thresh[i]]
            # print(captions[0], caption_sim)
            # print(above_thresh)

            # filter poor quality pseudo-labels
            # if logits[0] < pseudo_thresh:
                # continue

            scene_objects[scene_name]['object_feats'].append(visual_features)

            object_idx = len(scene_objects[scene_name]['object_feats']) - 1
            
            # get the class num
            # scene_objects[scene_name]['class_nums'].append(cls_num)

            # define if it is a top100 class or long tail class
            # if semantic_classes[cls_num] not in train_classes:
            #     flag = [True]

            flag = [False]
            captions = [caption_query[0]]
            text_features = [caption_feat[0].unsqueeze(0).cpu().numpy()]

            # iterate through the above_thresh to add the core concepts as captions
            for cls in above_thresh:
                captions.append(f'an image of a {cls}')
                text_features.append(core_feats[train_classes.index(cls)].unsqueeze(0).cpu().numpy())
                flag.append(True)

            # iterate through each caption
            for i in range(len(captions)): 
                caption = captions[i]  
                xfT = text_features[i] 
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
            
            # object_feats = np.stack(scene_objects[scene]['object_feats'])
            # object_feats = object_feats.squeeze(1)
            # pad with zeros to make object feats (max_objects, n)
            # object_feats = np.pad(object_feats, ((0, self.max_objects-object_feats.shape[0]), (0,0)), 'constant', constant_values=0)
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
        object_feats = np.stack(object_feats).squeeze(1)
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