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

def pad_text(num):
    if num<10:
        return '0000'+str(num)
    if num<100:
        return '000'+str(num)
           
    if num<1000:
        return '00'+str(num)
    
def load_model(device, model_path=None):
    # load zero-shot CLIP model
    model, _ = clip.load(name='ViT-B/32',
                         device=device,
                         loss_type='contrastive')
    if model_path is None:
        # Convert the dtype of parameters from float16 to float32
        for name, param in model.named_parameters():
            param.data = param.data.type(torch.float32)
    else:
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['state_dict'])
        for name, param in model.named_parameters():
            param.data = param.data.type(torch.float32)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model


class ScannetCaption(Dataset):
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
                 transform=None, target_transform=None,
                 ):

        # self.image_root = os.path.expanduser(image_root)
        # self.caption_root = os.path.expanduser(caption_root)
        device = 'cuda'
        CLIP_Net = load_model(device='cuda', model_path=None)
        CLIP_Net.eval()

        if isinstance(target_classes, str):
            with open(target_classes) as fin:
                _classes = [int(line.strip().split('_')[1]) - 1 for line in fin]
            target_classes = _classes
        

        # get the dataset splits
        splits_path = image_root+f'/data_download/complete_dataset/splits/nvs_sem_{split}.txt'
        # read each line and add to the splits file
        with open(splits_path) as fin:
            scenes = [line.strip() for line in fin]
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

        # define the target classes
        if split == 'train':
            target_classes = top100_classes
        else:
            target_classes = semantic_classes
        
        # remove some classes
        exclude_classes = ['ceiling', 'floor', 'wall']
        target_classes = [cls for cls in target_classes if cls not in exclude_classes]

        ###################### SAVE THE DATASET #####################3

        # store all info we may need for training models in the future
        # firstly, for each instance we save a hdf5 file containing
        # - the image name
        # - the bbox crop
        # - the captions
        # - the class number
        # - the visual features
        # - the text features
        # if the file already exists, we load it

        # check if caption_root exists
        if not os.path.exists(caption_root+f'/{split}'):
            # create the caption root directory
            os.makedirs(caption_root+f'/{split}')

            # get the scene paths
            scene_paths = image_root+f'/data_download/complete_dataset/data/'
            scene_n = 0
            for scene in tqdm.tqdm(scenes):
                instance_path = scene_paths+scene+'/iphone/instance/'
                # list all the images in instance_path
                images = glob.glob(instance_path+'*')
                images.sort()
                instance_data = {}
                for i in tqdm.tqdm(range(0,len(images),5)):
                # for i in range(0,len(images),5):
                    
                    # load the instance image
                    img_path = images[i]
                    instance_image = np.array(Image.open(img_path))
                    semantic_image = np.array(Image.open(img_path.replace('instance','label')))
                    rgb_path = img_path.replace('instance', 'rgb').replace('png','jpg')
                    rbg_image = Image.open(rgb_path).convert('RGB')
                    # get the unique instances
                    instances = np.unique(instance_image)
                    # iterate through each instance
                    for instance in instances:

                        _target = []

                        # get a list of the pixels that belong to this instance
                        mask = instance_image == instance

                        # get the bounding box
                        x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))

                        # get the label for this object
                        cls_num = int(np.unique(semantic_image[mask][0]))

                        caption = f'an image of a {semantic_classes[cls_num]}'

                        # skip if excluide
                        if semantic_classes[cls_num] in exclude_classes:
                            continue

                        # crop the image and tokenise the caption
                        crop = rbg_image.crop((x-20, y-20, x+w+20, y+h+20))

                        # print(f'an image of a {semantic_classes[cls_num]}')
                        # # display
                        # disp = np.array(crop)
                        # plt.imshow(disp)
                        # plt.show()

                        if transform is not None:
                            crop = transform(crop).unsqueeze(0)
                        if target_transform is not None:
                            target = target_transform(caption)
                            target = target
                        
                        # get the CLIP features
                        with torch.no_grad():
                            xfI, xfT = CLIP_Net(crop.to(device), target.to(device))

                        # create a new listing in the instances for this scene
                        if instance not in instance_data.keys():
                            instance_data[instance] = {'image_names': [], 'crops': [], 'captions': [], 'class_nums': [], 'visual_features': [], 'text_features': []}
                            # add the caption
                            instance_data[instance]['captions'].append(caption)
                            instance_data[instance]['text_features'].append(xfT.cpu())
                            instance_data[instance]['class_nums'].append(cls_num)

                        # update the variables for this instance
                        instance_data[instance]['image_names'].append(rgb_path)
                        instance_data[instance]['crops'].append([x, y, w, h])
                        instance_data[instance]['visual_features'].append(xfI.cpu())
                
                # then, we convert it into a dataset for the sampler
                for instance in instance_data.keys():
                    # open th h5py file
                    with h5py.File(caption_root+f'/{split}/{scene}_{instance}.h5', 'w') as f:
                        f.create_dataset('image_names', data=instance_data[instance]['image_names'])
                        f.create_dataset('crops', data=instance_data[instance]['crops'])
                        f.create_dataset('captions', data=instance_data[instance]['captions'])
                        f.create_dataset('class_nums', data=instance_data[instance]['class_nums'])
                        f.create_dataset('visual_features', data=instance_data[instance]['visual_features'])
                        f.create_dataset('text_features', data=instance_data[instance]['text_features'])

                scene_n+=1

                # if scene_n > 5:
                #     break

        ###################### PREPARE THE DATASET FOR TRAINING #####################

        # This can be editted dynamically as we change the training approach etc..

        # targets: list of (image_path, crop, caption) where multiple captions are generated for one image. The index of each target is used in the remaining variables
        # index_to_class: index to class number mapping. We will use the top 100 classes only for this dataset.
        # class_to_indices: class number to indices mapping. This is used for sampling.
        # class_to_img_indices: is not used
        targets = []
        index_to_class = {}
        class_to_indices = {}
        class_to_img_indices = {}
        scene_list = []
        idx = 0
        n_images = 0

        # list all files in the caption root
        caption_files = glob.glob(caption_root+f'/{split}/*.h5')
        # iterate through each file
        for caption_file in caption_files:
            scene_list.append(caption_file.split('/')[-1].split('_')[0])
            with h5py.File(caption_file, 'r') as f:
                image_names = f['image_names']
                crops = f['crops']
                visual_features = f['visual_features']

                captions = f['captions']
                class_nums = f['class_nums']
                text_features = f['text_features']

                # get the mean visual features
                xfI = np.mean(visual_features, axis=0)
                cls_num = class_nums[0]

                # check if the class is in the target classes
                if semantic_classes[cls_num] not in target_classes:
                    continue

                # iterate through each caption
                for i in range(len(captions)):
                    xfT = text_features[i]
                    # update the target
                    _target = [xfI, xfT]
                    _target.append(image_names[i])
                    _target.append(crops[i])
                    _target.append(captions[i])
                    
                    # update the classes
                    index_to_class[idx] = cls_num
                    class_to_indices.setdefault(cls_num, []).append(idx)
                    
                    idx += 1
                    n_images+=1 
                    targets.append(_target)

        self.targets = targets
        self.scenes = scene_list
        self.target_classes = target_classes
        self.index_to_class = index_to_class
        self.class_to_indices = class_to_indices
        self.class_to_img_indices = class_to_img_indices

        self.n_images = n_images

        self.transform = transform
        self.target_transform = target_transform

    # def __getitem__(self, index):
    #     # this is just an image-text pair
    #     img_path, crop, target = self.targets[index]

    #     # we open the image and apply the necessary transformations
    #     img = Image.open(img_path).convert('RGB')
    #     # crop the image
    #     img = img.crop((crop[0]-20, crop[1]-20, crop[0]+crop[2]+20, crop[1]+crop[3]+20))
    #     if self.transform is not None:
    #         img = self.transform(img)
    #     # the target transformation is just tokenisation
    #     if self.target_transform is not None:
    #         target = self.target_transform(target)
    #         target = target.squeeze(0)
        
    #     return img, target, self.index_to_class[index], index

    def __getitem__(self, index):
        # this is just an image-text pair
        xfI, xfT, img_name, crop, caption = self.targets[index]
        
        return xfI, xfT, self.index_to_class[index], index, self.scenes[index], img_name, crop, caption

    def __len__(self):
        return len(self.targets)


class ScannetSampler(Sampler):
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


class ScannetSceneQuery(Dataset):
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
                 transform=None, target_transform=None, seen_classes='top100', use_affordances = False
                 ):

        # self.image_root = os.path.expanduser(image_root)
        # self.caption_root = os.path.expanduser(caption_root)
        device = 'cuda'
        CLIP_Net = load_model(device='cuda', model_path=None)
        CLIP_Net.eval()

        if isinstance(target_classes, str):
            with open(target_classes) as fin:
                _classes = [int(line.strip().split('_')[1]) - 1 for line in fin]
            target_classes = _classes
        

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

        # open the unambiguous affordances as a json file
        with open(image_root+f'/data_download/complete_dataset/metadata/unambiguous_affordances.json') as fin:
            unambiguous_affordances = json.load(fin)

        affordance_features = {}
        for affordance, object_top100 in unambiguous_affordances.items():
            with torch.no_grad():
                caption_query = [f'an image of an {affordance}']
                caption_targets = target_transform(caption_query)
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

        # define the target classes
        if split == 'train':
            # target_classes = top100_classes
            target_classes = train_classes
            target_scenes = scenes[0:50]
        elif split == 'calib':
            target_classes = calib_classes
            target_scenes = scenes[0:50]
            # scenes = scenes[-50:]
        else:
            target_classes = test_classes
            target_scenes = scenes
        
        # remove some classes
        exclude_classes = ['ceiling', 'floor', 'wall']
        target_classes = [cls for cls in target_classes if cls not in exclude_classes]

        self.target_classes = target_classes

        ###################### SAVE THE DATASET #####################3

        # store all info we may need for training models in the future
        # firstly, for each instance we save a hdf5 file containing
        # - the image name
        # - the bbox crop
        # - the captions
        # - the class number
        # - the visual features
        # - the text features
        # if the file already exists, we load it

        # check if caption_root exists
        if not os.path.exists(caption_root+f'/{dataset_split}'):
            # create the caption root directory
            os.makedirs(caption_root+f'/{dataset_split}')

            # get the scene paths
            scene_paths = image_root+f'/data_download/complete_dataset/data/'
            scene_n = 0
            for scene in tqdm.tqdm(scenes):
                instance_path = scene_paths+scene+'/iphone/instance/'
                # list all the images in instance_path
                images = glob.glob(instance_path+'*')
                images.sort()
                instance_data = {}
                for i in tqdm.tqdm(range(0,len(images),5)):
                # for i in range(0,len(images),5):
                    
                    # load the instance image
                    img_path = images[i]
                    instance_image = np.array(Image.open(img_path))
                    semantic_image = np.array(Image.open(img_path.replace('instance','label')))
                    rgb_path = img_path.replace('instance', 'rgb').replace('png','jpg')
                    rbg_image = Image.open(rgb_path).convert('RGB')
                    # get the unique instances
                    instances = np.unique(instance_image)
                    # iterate through each instance
                    for instance in instances:

                        _target = []

                        # get a list of the pixels that belong to this instance
                        mask = instance_image == instance

                        # get the bounding box
                        x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))

                        # get the label for this object
                        cls_num = int(np.unique(semantic_image[mask][0]))

                        caption = f'an image of a {semantic_classes[cls_num]}'

                        # skip if excluide
                        if semantic_classes[cls_num] in exclude_classes:
                            continue

                        # crop the image and tokenise the caption
                        crop = rbg_image.crop((x-20, y-20, x+w+20, y+h+20))

                        # print(f'an image of a {semantic_classes[cls_num]}')
                        # # display
                        # disp = np.array(crop)
                        # plt.imshow(disp)
                        # plt.show()

                        if transform is not None:
                            crop = transform(crop).unsqueeze(0)
                        if target_transform is not None:
                            target = target_transform(caption)
                            target = target
                        
                        # get the CLIP features
                        with torch.no_grad():
                            xfI, xfT = CLIP_Net(crop.to(device), target.to(device))

                        # create a new listing in the instances for this scene
                        if instance not in instance_data.keys():
                            instance_data[instance] = {'image_names': [], 'crops': [], 'captions': [], 'class_nums': [], 'visual_features': [], 'text_features': []}
                            # add the caption
                            instance_data[instance]['captions'].append(caption)
                            instance_data[instance]['text_features'].append(xfT.cpu())
                            instance_data[instance]['class_nums'].append(cls_num)

                        # update the variables for this instance
                        instance_data[instance]['image_names'].append(rgb_path)
                        instance_data[instance]['crops'].append([x, y, w, h])
                        instance_data[instance]['visual_features'].append(xfI.cpu())
                
                # then, we convert it into a dataset for the sampler
                for instance in instance_data.keys():
                    # open th h5py file
                    with h5py.File(caption_root+f'/{dataset_split}/{scene}_{instance}.h5', 'w') as f:
                        f.create_dataset('image_names', data=instance_data[instance]['image_names'])
                        f.create_dataset('crops', data=instance_data[instance]['crops'])
                        f.create_dataset('captions', data=instance_data[instance]['captions'])
                        f.create_dataset('class_nums', data=instance_data[instance]['class_nums'])
                        f.create_dataset('visual_features', data=instance_data[instance]['visual_features'])
                        f.create_dataset('text_features', data=instance_data[instance]['text_features'])

                scene_n+=1

                # if scene_n > 5:
                #     break

        ###################### PREPARE THE DATASET FOR TRAINING #####################

        # This can be editted dynamically as we change the training approach etc..

        scene_objects = {}
        scene_queries = {}
        # list all files in the caption root
        caption_files = glob.glob(caption_root+f'/{dataset_split}/*.h5')
        # iterate through each file
        for caption_file in tqdm.tqdm(caption_files):
            scene_name = caption_file.split('/')[-1].split('_')[0]

            if scene_name not in target_scenes:
                continue

            if scene_name not in scene_objects.keys():
                scene_objects[scene_name] = {
                    'object_feats': [],
                    'class_nums': [],
                    'object_file': []
                }
                scene_queries[scene_name] = {
                    'captions': [],
                    'query_feats': [],
                    'object_ids': [],
                    'unseen': []
                }
            
            with h5py.File(caption_file, 'r') as f:
                image_names = f['image_names']
                crops = f['crops']
                visual_features = f['visual_features']

                captions = f['captions']
                class_nums = f['class_nums']
                text_features = f['text_features']

                cls_num = class_nums[0]
                if split == 'train':
                # if split == 'train' or split == 'calib':
                    # only add the object if it is in the target classes
                    if semantic_classes[cls_num] not in target_classes:
                        continue

                # get the mean visual features
                xfI = np.mean(visual_features, axis=0)
                scene_objects[scene_name]['object_feats'].append(xfI)
                object_idx = len(scene_objects[scene_name]['object_feats']) - 1

                # get the class num
                scene_objects[scene_name]['class_nums'].append(cls_num)

                # get the object file
                scene_objects[scene_name]['object_file'].append(caption_file)

                # define if it is a top100 class or long tail class
                flag = [False]
                if semantic_classes[cls_num] not in train_classes:
                    flag = [True]

                # only add the object if it is in the target classes
                if semantic_classes[cls_num] not in target_classes:
                    continue
                
                if use_affordances:
                    if semantic_classes[cls_num] in affordance_features.keys():
                        # print(semantic_classes[cls_num])
                        # print(affordance_features[semantic_classes[cls_num]].shape)

                        aff_feat = affordance_features[semantic_classes[cls_num]].unsqueeze(0).cpu().numpy()
                        text_features = np.concatenate((text_features, aff_feat), axis=0)

                        captions = [captions[0].decode('utf-8'), f'an image of {unambiguous_affordances[semantic_classes[cls_num]]}']

                        flag = [False, True]
                    else:
                        flag = [False]
                        captions = [captions[0].decode('utf-8')]

                # iterate through each caption
                for i in range(len(captions)):
                    xfT = text_features[i]
                    caption = captions[i]
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
        max_objects = max([len(scene_objects[scene]['object_feats']) for scene in scene_objects.keys()])

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
            object_feats = np.stack(scene_objects[scene]['object_feats']).squeeze(1)
            # pad with zeros to make object feats (max_objects, 512)
            object_feats = np.pad(object_feats, ((0, max_objects-object_feats.shape[0]), (0,0)), 'constant', constant_values=0)

            # prepare the object files
            object_files = scene_objects[scene]['object_file']
            # pad with None to make object files (max_objects)
            object_files = object_files + ['None']*(max_objects-len(object_files))

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
        
        return self.query_feats[index], self.query_labels[index], self.captions[index], self.object_feats[index], self.unseen[index], self.object_files[index]
    
    def __len__(self):
        return len(self.captions)
    
class ScannetPseudoLabel(Dataset):
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
                 transform=None, target_transform=None, pseudo_thresh=0.3, seen_classes='top100', pseudo_method='scene',
                 ):

        print(pseudo_method, pseudo_thresh)
        
        # self.image_root = os.path.expanduser(image_root)
        # self.caption_root = os.path.expanduser(caption_root)
        device = 'cuda'
        CLIP_Net = load_model(device='cuda', model_path=None)
        CLIP_Net.eval()

        if isinstance(target_classes, str):
            with open(target_classes) as fin:
                _classes = [int(line.strip().split('_')[1]) - 1 for line in fin]
            target_classes = _classes
        

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

        # define the target classes
        if split == 'train':
            # target_classes = top100_classes
            target_classes = train_classes
            target_scenes = scenes[0:50]
        elif split == 'calib':
            target_classes = calib_classes
            target_scenes = scenes[0:50]
            # scenes = scenes[-50:]
        else:
            target_classes = test_classes
            target_scenes = scenes
        
        # remove some classes
        exclude_classes = ['ceiling', 'floor', 'wall']
        target_classes = [cls for cls in target_classes if cls not in exclude_classes]

        self.target_classes = target_classes

        ###################### SAVE THE DATASET #####################3

        # store all info we may need for training models in the future
        # firstly, for each instance we save a hdf5 file containing
        # - the image name
        # - the bbox crop
        # - the captions
        # - the class number
        # - the visual features
        # - the text features
        # if the file already exists, we load it

        # check if caption_root exists
        if not os.path.exists(caption_root+f'/{dataset_split}'):
            # create the caption root directory
            os.makedirs(caption_root+f'/{dataset_split}')

            # get the scene paths
            scene_paths = image_root+f'/data_download/complete_dataset/data/'
            scene_n = 0
            for scene in tqdm.tqdm(scenes):
                instance_path = scene_paths+scene+'/iphone/instance/'
                # list all the images in instance_path
                images = glob.glob(instance_path+'*')
                images.sort()
                instance_data = {}
                for i in tqdm.tqdm(range(0,len(images),5)):
                # for i in range(0,len(images),5):
                    
                    # load the instance image
                    img_path = images[i]
                    instance_image = np.array(Image.open(img_path))
                    semantic_image = np.array(Image.open(img_path.replace('instance','label')))
                    rgb_path = img_path.replace('instance', 'rgb').replace('png','jpg')
                    rbg_image = Image.open(rgb_path).convert('RGB')
                    # get the unique instances
                    instances = np.unique(instance_image)
                    # iterate through each instance
                    for instance in instances:

                        _target = []

                        # get a list of the pixels that belong to this instance
                        mask = instance_image == instance

                        # get the bounding box
                        x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))

                        # get the label for this object
                        cls_num = int(np.unique(semantic_image[mask][0]))

                        caption = f'an image of a {semantic_classes[cls_num]}'

                        # skip if excluide
                        if semantic_classes[cls_num] in exclude_classes:
                            continue

                        # crop the image and tokenise the caption
                        crop = rbg_image.crop((x-20, y-20, x+w+20, y+h+20))

                        # print(f'an image of a {semantic_classes[cls_num]}')
                        # # display
                        # disp = np.array(crop)
                        # plt.imshow(disp)
                        # plt.show()

                        if transform is not None:
                            crop = transform(crop).unsqueeze(0)
                        if target_transform is not None:
                            target = target_transform(caption)
                            target = target
                        
                        # get the CLIP features
                        with torch.no_grad():
                            xfI, xfT = CLIP_Net(crop.to(device), target.to(device))

                        # create a new listing in the instances for this scene
                        if instance not in instance_data.keys():
                            instance_data[instance] = {'image_names': [], 'crops': [], 'captions': [], 'class_nums': [], 'visual_features': [], 'text_features': []}
                            # add the caption
                            instance_data[instance]['captions'].append(caption)
                            instance_data[instance]['text_features'].append(xfT.cpu())
                            instance_data[instance]['class_nums'].append(cls_num)

                        # update the variables for this instance
                        instance_data[instance]['image_names'].append(rgb_path)
                        instance_data[instance]['crops'].append([x, y, w, h])
                        instance_data[instance]['visual_features'].append(xfI.cpu())
                
                # then, we convert it into a dataset for the sampler
                for instance in instance_data.keys():
                    # open th h5py file
                    with h5py.File(caption_root+f'/{dataset_split}/{scene}_{instance}.h5', 'w') as f:
                        f.create_dataset('image_names', data=instance_data[instance]['image_names'])
                        f.create_dataset('crops', data=instance_data[instance]['crops'])
                        f.create_dataset('captions', data=instance_data[instance]['captions'])
                        f.create_dataset('class_nums', data=instance_data[instance]['class_nums'])
                        f.create_dataset('visual_features', data=instance_data[instance]['visual_features'])
                        f.create_dataset('text_features', data=instance_data[instance]['text_features'])

                scene_n+=1

                # if scene_n > 5:
                #     break

        ###################### PREPARE THE DATASET FOR TRAINING #####################

        # pseudo-label queries for the  unlabelled scenes
        with torch.no_grad():
            # either do all classes, or just the unseen classes
            # unseen_classes = [cls for cls in semantic_classes if cls not in target_classes]
            unseen_classes = [cls for cls in semantic_classes]
            # remove exclude classes
            unseen_classes = [cls for cls in unseen_classes if cls not in exclude_classes]
            unseen_queries = [f'an image of a {cls}' for cls in unseen_classes]
            unseen_targets = target_transform(unseen_queries)
            unseen_targets = unseen_targets.to(device)
            unseen_feats = CLIP_Net.encode_text(unseen_targets)

            # get the "other" class embedding
            other_target = target_transform(['an image of an object'])
            other_target = other_target.to(device)
            other_feat = CLIP_Net.encode_text(other_target)
                            
        # This can be editted dynamically as we change the training approach etc..
        scene_objects = {}
        scene_queries = {}
        pseudo_objects = {}
        # list all files in the caption root
        caption_files = glob.glob(caption_root+f'/{dataset_split}/*.h5')
        # iterate through each file
        for caption_file in tqdm.tqdm(caption_files):
            scene_name = caption_file.split('/')[-1].split('_')[0]

            # if scene_name not in scenes:
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
            
            with h5py.File(caption_file, 'r') as f:
                image_names = f['image_names']
                crops = f['crops']
                visual_features = f['visual_features']

                captions = f['captions']
                class_nums = f['class_nums']
                text_features = f['text_features']

                cls_num = class_nums[0]
                # if split == 'train':
                # if split == 'train' or split == 'calib':
                
                # get the mean visual features
                xfI = np.mean(visual_features, axis=0)

                # if a labelled scene, we skip all objects that are not in the target classes
                if scene_name in target_scenes:
                    if semantic_classes[cls_num] not in target_classes:
                        continue
                # otherwise, we attempt to pseudo-label the object
                else:    
                    # # get similarity between the text features and the unseen features
                    # img_embedding = torch.tensor(xfI).to(device)
                    # unseen_feats_norm = unseen_feats / unseen_feats.norm(dim=-1, keepdim=True)
                    # img_embedding = img_embedding / img_embedding.norm(dim=-1, keepdim=True)
                    # logits = torch.matmul(unseen_feats_norm, img_embedding.T).squeeze(0)

                    # # get the max logit and index
                    # max_logit, max_idx = torch.max(logits, 0)

                    # if max_logit > pseudo_thresh:
                    #     pseudo_labels = [unseen_queries[max_idx]]
                    #     pseudo_features = [unseen_feats[max_idx]]
                    # else:
                    #     pseudo_labels = ["an image of an object"]
                    #     pseudo_features = [other_feat]
                    #     continue
                    
                    # # get the indices of all the classes that are above the threshold
                    # # pseudo_labels = [unseen_queries[i] for i in range(len(logits)) if logits[i] > pseudo_thresh]
                    # # pseudo_features = [unseen_feats[i] for i in range(len(logits)) if logits[i] > pseudo_thresh]

                    # # assign to captions
                    # captions = pseudo_labels
                    # text_features = [f.cpu().numpy() for f in pseudo_features]

                    # create a new listing in the instances for this scene
                    if scene_name not in pseudo_objects.keys():
                        pseudo_objects[scene_name] = {
                            'object_feats': [],
                            'query_feats': [],
                            'class_nums': [],
                            'captions': [],
                        }

                    # append object and query to pseudo objects
                    pseudo_objects[scene_name]['object_feats'].append(xfI)
                    pseudo_objects[scene_name]['class_nums'].append(cls_num)
                    # check if caption in the data structure
                    if captions[0] not in pseudo_objects[scene_name]['captions']:
                        pseudo_objects[scene_name]['query_feats'].append(text_features[0])
                        pseudo_objects[scene_name]['captions'].append(captions[0])
                    continue

                # append object
                scene_objects[scene_name]['object_feats'].append(xfI)

                object_idx = len(scene_objects[scene_name]['object_feats']) - 1
                
                # get the class num
                scene_objects[scene_name]['class_nums'].append(cls_num)

                # define if it is a top100 class or long tail class
                unseen = False
                if semantic_classes[cls_num] not in train_classes:
                    unseen = True

                # only add the object if it is in the target classes
                # if semantic_classes[cls_num] not in target_classes:
                #     continue

                # iterate through each caption
                for i in range(len(captions)):
                    xfT = text_features[i]
                    caption = captions[i]

                    # check if caption not in the data structure
                    if caption not in scene_queries[scene_name]['captions']:
                        scene_queries[scene_name]['captions'].append(caption)
                        scene_queries[scene_name]['query_feats'].append(xfT)
                        scene_queries[scene_name]['object_ids'].append([str(object_idx)])
                        scene_queries[scene_name]['unseen'].append(unseen)
                    else:
                        scene_queries[scene_name]['object_ids'][scene_queries[scene_name]['captions'].index(caption)].append(str(object_idx))

        # check for pseudo-label objects
        for scene in pseudo_objects.keys():
            # get object and query features
            object_feats = np.stack(pseudo_objects[scene]['object_feats']).squeeze(1)
            query_feats = np.stack(pseudo_objects[scene]['query_feats']).squeeze(1)

            # convert to tensor and add to device
            object_feats = torch.tensor(object_feats).to(device)
            query_feats = torch.tensor(query_feats).to(device)

            # normalise and get similarity
            object_feats_norm = object_feats / object_feats.norm(dim=-1, keepdim=True)
            query_feats_norm = query_feats / query_feats.norm(dim=-1, keepdim=True)
            logits = torch.matmul(query_feats_norm, object_feats_norm.T).squeeze(0)

            # get the max logit and index for each object
            max_logit, max_idx = torch.max(logits, 0)

            # iterate through each max_logit and max_idx
            for k in range(len(max_logit)):
                # get the object of interest
                pseudo_object = pseudo_objects[scene]['object_feats'][k]
                pseudo_cls_num = pseudo_objects[scene]['class_nums'][k]
                # use only scene queries
                if pseudo_method == 'scene':
                    # check if there is a match in the scene
                    if max_logit[k] > pseudo_thresh:
                        pseudo_caption = pseudo_objects[scene]['captions'][max_idx[k]]
                        pseudo_query = pseudo_objects[scene]['query_feats'][max_idx[k]]
                    else:
                        continue
                # otherwise, check for a match with other objects
                elif pseudo_method == 'all':
                    # get similarity between the text features and the unseen features
                    unseen_feats_norm = unseen_feats / unseen_feats.norm(dim=-1, keepdim=True)
                    pseudo_object_norm = torch.tensor(pseudo_object).to(device)
                    pseudo_object_norm = pseudo_object_norm / pseudo_object_norm.norm(dim=-1, keepdim=True)
                    logits = torch.matmul(unseen_feats_norm, pseudo_object_norm.T).squeeze(0)

                    # get the max logit and index
                    unseen_max, unseen_idx = torch.max(logits, 0)

                    if unseen_max > pseudo_thresh:
                        pseudo_caption = unseen_queries[unseen_idx]
                        pseudo_query = unseen_feats[unseen_idx].cpu().numpy()
                    else:
                        continue

                # append object
                scene_objects[scene]['object_feats'].append(pseudo_object)

                pseudo_idx = len(scene_objects[scene]['object_feats']) - 1
                unseen = True

                # print(semantic_classes[pseudo_cls_num], pseudo_caption)

                # add to datastructure
                if pseudo_caption not in scene_queries[scene]['captions']:
                    scene_queries[scene]['captions'].append(pseudo_caption)
                    scene_queries[scene]['query_feats'].append(pseudo_query)
                    scene_queries[scene]['object_ids'].append([str(pseudo_idx)])
                    scene_queries[scene]['unseen'].append(unseen)
                else:
                    scene_queries[scene]['object_ids'][scene_queries[scene]['captions'].index(pseudo_caption)].append(str(pseudo_idx))

        # query_feats: text feature of unique class in the scene
        # captions: captions of unique class in the scene for explainability
        # object_feats: visual feature of all unique object instances
        # query_labels: 0 or 1 for each object instance

        # get the maximum number of objects in a single scene, to pad the object features
        max_objects = max([len(scene_objects[scene]['object_feats']) for scene in scene_objects.keys()])

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
            object_feats = np.stack(scene_objects[scene]['object_feats']).squeeze(1)
            # pad with zeros to make object feats (max_objects, 512)
            object_feats = np.pad(object_feats, ((0, max_objects-object_feats.shape[0]), (0,0)), 'constant', constant_values=0)

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

class ScannetCaptionPseudoLabel(Dataset):
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
                 transform=None, target_transform=None, pseudo_thresh=0.3, seen_classes='top100', pseudo_method='scene', use_affordances=False,
                 ):

        print(pseudo_method, pseudo_thresh)
        
        # self.image_root = os.path.expanduser(image_root)
        # self.caption_root = os.path.expanduser(caption_root)
        device = 'cuda'
        CLIP_Net = load_model(device='cuda', model_path=None)
        CLIP_Net.eval()

        if isinstance(target_classes, str):
            with open(target_classes) as fin:
                _classes = [int(line.strip().split('_')[1]) - 1 for line in fin]
            target_classes = _classes
        

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

        # define the target classes
        if split == 'train':
            # target_classes = top100_classes
            target_classes = train_classes
            target_scenes = [] # scenes[0:50]
        elif split == 'calib':
            target_classes = calib_classes
            target_scenes = [] #scenes[0:50]
            # scenes = scenes[-50:]
        else:
            target_classes = test_classes
            target_scenes = scenes
        
        # remove some classes
        exclude_classes = ['ceiling', 'floor', 'wall']
        target_classes = [cls for cls in target_classes if cls not in exclude_classes]

        self.target_classes = target_classes

        ###################### PREPARE THE DATASET FOR TRAINING #####################

        # pseudo-label queries for the  unlabelled scenes
        with torch.no_grad():
            # either do all classes, or just the unseen classes
            # unseen_classes = [cls for cls in semantic_classes if cls not in target_classes]
            unseen_classes = [cls for cls in semantic_classes]
            # remove exclude classes
            unseen_classes = [cls for cls in unseen_classes if cls not in exclude_classes]
            unseen_queries = [f'an image of a {cls}' for cls in unseen_classes]
            unseen_targets = target_transform(unseen_queries)
            unseen_targets = unseen_targets.to(device)
            unseen_feats = CLIP_Net.encode_text(unseen_targets)

            # get the "other" class embedding
            other_target = target_transform(['an image of an object'])
            other_target = other_target.to(device)
            other_feat = CLIP_Net.encode_text(other_target)
                            
        # This can be editted dynamically as we change the training approach etc..
        scene_objects = {}
        scene_queries = {}
        pseudo_objects = {}
        # list all files in the caption root
        caption_files = glob.glob(caption_root+f'/{dataset_split}_caption/*.h5')
        # iterate through each file
        for caption_file in tqdm.tqdm(caption_files):
            scene_name = caption_file.split('/')[-1].split('_')[0]

            # if scene_name not in scenes:
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
            
            # attempt to open the caption data
            with h5py.File(caption_file, 'r') as f: 
                # print(f['captions'][()].decode('utf-8'))
                try:
                    object_tag = f['object_tag'][()].decode('utf-8').replace('_', ' ')
                    object_description = f['object_description'][()].decode('utf-8').replace('_', ' ')
                    object_affordance = f['object_affordance'][()].decode('utf-8').replace('_', ' ')
                    # ignore invalid
                    if object_tag == 'invalid':
                        continue
                    # print(object_tag, object_description, object_affordance)
                except KeyError:
                    continue
            
            # open the object info file
            instance_file = caption_file.replace('train_caption', 'train')
            with h5py.File(instance_file, 'r') as f:
                image_names = f['image_names']
                crops = f['crops']
                visual_features = f['visual_features']

                captions = f['captions']
                class_nums = f['class_nums']
                text_features = f['text_features']

                cls_num = class_nums[0]
                # if split == 'train':
                # if split == 'train' or split == 'calib':
                
                # get the mean visual features
                xfI = np.mean(visual_features, axis=0)

                # if a labelled scene, we skip all objects that are not in the target classes
                if scene_name in target_scenes:
                    # ignore if not in target classes
                    if semantic_classes[cls_num] not in target_classes:
                        continue
                    # append object
                    else:
                        scene_objects[scene_name]['object_feats'].append(xfI)

                        object_idx = len(scene_objects[scene_name]['object_feats']) - 1
                        
                        # get the class num
                        scene_objects[scene_name]['class_nums'].append(cls_num)

                        # define if it is a top100 class or long tail class
                        unseen = False
                        if semantic_classes[cls_num] not in train_classes:
                            unseen = True

                        # only add the object if it is in the target classes
                        # if semantic_classes[cls_num] not in target_classes:
                        #     continue

                        # iterate through each caption
                        for i in range(len(captions)):
                            xfT = text_features[i]
                            caption = captions[i]

                            # check if caption not in the data structure
                            if caption not in scene_queries[scene_name]['captions']:
                                scene_queries[scene_name]['captions'].append(caption)
                                scene_queries[scene_name]['query_feats'].append(xfT)
                                scene_queries[scene_name]['object_ids'].append([str(object_idx)])
                                scene_queries[scene_name]['unseen'].append(unseen)
                            else:
                                scene_queries[scene_name]['object_ids'][scene_queries[scene_name]['captions'].index(caption)].append(str(object_idx))
                                
                # otherwise, we attempt to pseudo-label the object
                else:  
                    captions = [object_tag, object_description, object_affordance]
                    # pseudo-label queries for the  unlabelled scenes
                    with torch.no_grad():
                        caption_query = [f'an image of a {captions[0]}', f'an image of a {captions[1]}', f'an image of an object for {captions[2]}']
                        caption_targets = target_transform(caption_query)
                        caption_targets = caption_targets.to(device)
                        caption_feat = CLIP_Net.encode_text(caption_targets)

                    # get similarity between the object features and caption feature
                    img_embedding = torch.tensor(xfI).to(device)
                    caption_embedding = torch.tensor(caption_feat).to(device)
                    img_embedding = img_embedding / img_embedding.norm(dim=-1, keepdim=True)
                    caption_embedding = caption_embedding / caption_embedding.norm(dim=-1, keepdim=True)
                    logits = torch.matmul(caption_embedding, img_embedding.T).squeeze(0)
                    # print(captions[0], semantic_classes[cls_num], logits)

                    # filter poor qualiy pseudo-labels
                    if logits[0] < pseudo_thresh:
                        continue

                    scene_objects[scene_name]['object_feats'].append(xfI)

                    object_idx = len(scene_objects[scene_name]['object_feats']) - 1
                    
                    # get the class num
                    scene_objects[scene_name]['class_nums'].append(cls_num)

                    # define if it is a top100 class or long tail class
                    flag = [False]
                    if semantic_classes[cls_num] not in train_classes:
                        flag = [True]

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
                                
                    # # create a new listing in the instances for this scene
                    # if scene_name not in pseudo_objects.keys():
                    #     pseudo_objects[scene_name] = {
                    #         'object_feats': [],
                    #         'query_feats': [],
                    #         'class_nums': [],
                    #         'captions': [],
                    #     }

                    # # append object and query to pseudo objects
                    # pseudo_objects[scene_name]['object_feats'].append(xfI)
                    # pseudo_objects[scene_name]['class_nums'].append(cls_num)
                    # # check if caption in the data structure
                    # if captions[0] not in pseudo_objects[scene_name]['captions']:
                    #     pseudo_objects[scene_name]['query_feats'].append(text_features[0])
                    #     pseudo_objects[scene_name]['captions'].append(captions[0])
                    # continue

        # # check for pseudo-label objects
        # for scene in pseudo_objects.keys():
        #     # get object and query features
        #     object_feats = np.stack(pseudo_objects[scene]['object_feats']).squeeze(1)
        #     query_feats = np.stack(pseudo_objects[scene]['query_feats']).squeeze(1)

        #     # convert to tensor and add to device
        #     object_feats = torch.tensor(object_feats).to(device)
        #     query_feats = torch.tensor(query_feats).to(device)

        #     # normalise and get similarity
        #     object_feats_norm = object_feats / object_feats.norm(dim=-1, keepdim=True)
        #     query_feats_norm = query_feats / query_feats.norm(dim=-1, keepdim=True)
        #     logits = torch.matmul(query_feats_norm, object_feats_norm.T).squeeze(0)

        #     # get the max logit and index for each object
        #     max_logit, max_idx = torch.max(logits, 0)

        #     # iterate through each max_logit and max_idx
        #     for k in range(len(max_logit)):
        #         # get the object of interest
        #         pseudo_object = pseudo_objects[scene]['object_feats'][k]
        #         pseudo_cls_num = pseudo_objects[scene]['class_nums'][k]
        #         # use only scene queries
        #         if pseudo_method == 'scene':
        #             # check if there is a match in the scene
        #             if max_logit[k] > pseudo_thresh:
        #                 pseudo_caption = pseudo_objects[scene]['captions'][max_idx[k]]
        #                 pseudo_query = pseudo_objects[scene]['query_feats'][max_idx[k]]
        #             else:
        #                 continue
        #         # otherwise, check for a match with other objects
        #         elif pseudo_method == 'all':
        #             # get similarity between the text features and the unseen features
        #             unseen_feats_norm = unseen_feats / unseen_feats.norm(dim=-1, keepdim=True)
        #             pseudo_object_norm = torch.tensor(pseudo_object).to(device)
        #             pseudo_object_norm = pseudo_object_norm / pseudo_object_norm.norm(dim=-1, keepdim=True)
        #             logits = torch.matmul(unseen_feats_norm, pseudo_object_norm.T).squeeze(0)

        #             # get the max logit and index
        #             unseen_max, unseen_idx = torch.max(logits, 0)

        #             if unseen_max > pseudo_thresh:
        #                 pseudo_caption = unseen_queries[unseen_idx]
        #                 pseudo_query = unseen_feats[unseen_idx].cpu().numpy()
        #             else:
        #                 continue

        #         # append object
        #         scene_objects[scene]['object_feats'].append(pseudo_object)

        #         pseudo_idx = len(scene_objects[scene]['object_feats']) - 1
        #         unseen = True

        #         # print(semantic_classes[pseudo_cls_num], pseudo_caption)

        #         # add to datastructure
        #         if pseudo_caption not in scene_queries[scene]['captions']:
        #             scene_queries[scene]['captions'].append(pseudo_caption)
        #             scene_queries[scene]['query_feats'].append(pseudo_query)
        #             scene_queries[scene]['object_ids'].append([str(pseudo_idx)])
        #             scene_queries[scene]['unseen'].append(unseen)
        #         else:
        #             scene_queries[scene]['object_ids'][scene_queries[scene]['captions'].index(pseudo_caption)].append(str(pseudo_idx))

        # query_feats: text feature of unique class in the scene
        # captions: captions of unique class in the scene for explainability
        # object_feats: visual feature of all unique object instances
        # query_labels: 0 or 1 for each object instance

        # get the maximum number of objects in a single scene, to pad the object features
        max_objects = max([len(scene_objects[scene]['object_feats']) for scene in scene_objects.keys()])

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
            object_feats = np.stack(scene_objects[scene]['object_feats']).squeeze(1)
            # pad with zeros to make object feats (max_objects, 512)
            object_feats = np.pad(object_feats, ((0, max_objects-object_feats.shape[0]), (0,0)), 'constant', constant_values=0)

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
        
        return self.query_feats[index], self.query_labels[index], self.captions[index], self.object_feats[index], self.unseen[index]
    
    def __len__(self):
        return len(self.captions)