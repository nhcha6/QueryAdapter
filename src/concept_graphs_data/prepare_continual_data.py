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
import transformers
import os
import h5py

def get_mapped_objects(caption_root, dataset_split):
    mapped_objects = {'feature': [], 'captions': [], 'scene_name': [], 'descriptions': [], 'core_classes': []}
    # list all files in the caption root
    instance_files = glob.glob(caption_root+f'/{dataset_split}/*.h5')
    # iterate through each file
    count = 0
    for instance_file in tqdm.tqdm(instance_files):
        # get the scene_name
        scene_name = instance_file.split('/')[-1].split('_')[0]

        # # stop early
        # if count > 1000:
        #     break
        # count += 1
        
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
                continue
                
        # iterate through each caption
        mapped_objects['feature'].append(torch.tensor(xfI))
        mapped_objects['descriptions'].append(object_description)
        mapped_objects['captions'].append(object_tag)
        mapped_objects['scene_name'].append(scene_name)
        mapped_objects['core_classes'].append([])

    return mapped_objects

dataset_root = '/home/nicolas/hpc-home/Datasets/scannetpp/data_download/complete_dataset'
caption_root = f'{dataset_root}/finetune_data/conceptgraph'
dataset_split = 'train'

# open the splits
splits_path = f'{dataset_root}/splits/nvs_sem_{dataset_split}.txt'
# read each line and add to the splits file
with open(splits_path) as fin:
    scenes = [line.strip() for line in fin]
# scenes = scenes[0:2]

# open the list of semantic classes
semantic_fp = f'{dataset_root}/metadata/semantic_classes.txt'
with open(semantic_fp) as fin:
    semantic_classes = [line.strip() for line in fin]
# open the top 100 classses
top_fp = f'{dataset_root}/metadata/semantic_benchmark/top100.txt'
with open(top_fp) as fin:
    top100 = [line.strip() for line in fin]
# get the top 50 classes
exclude_classes = ['wall', 'floor', 'ceiling']
target_classes = [c for c in semantic_classes if c not in exclude_classes]
top50  = [c for c in top100[0:53] if c not in exclude_classes]
long_tail = [c for c in target_classes if c not in top50]

# import clip
device = 'cuda'
CLIP_Net, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-H-14", "laion2b_s32b_b79k"
)
CLIP_Net = CLIP_Net.to(device)
CLIP_Net.eval()
clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

# # open the unambiguous affordances as a json file
# affordance_fp = f'{dataset_root}/metadata/unambiguous_affordances.json'
# with open(affordance_fp) as fin:
#     unambiguous_affordances = json.load(fin)
# # process the affordances
# affordance_features = {}
# for affordance, object_top100 in unambiguous_affordances.items():
#     with torch.no_grad():
#         caption_query = [f'an image of {affordance}']
#         caption_targets = clip_tokenizer(caption_query)
#         caption_targets = caption_targets.to(device)
#         caption_feat = CLIP_Net.encode_text(caption_targets)
#         affordance_features[object_top100] = caption_feat
# unambiguous_affordances = {v: k for k, v in unambiguous_affordances.items()}
# affordance_list = list(unambiguous_affordances.values())
# # sort the list for consistency
# affordance_list.sort()

# get the descriptive queries
affordance_fp = f'{dataset_root}/metadata/top100_affordances.txt'
with open(affordance_fp) as fin:
    top100_affordances = [line.strip() for line in fin]
affordance_queries = {}
affordance_features = {}
affordance_list = []
for idx, cls in enumerate(top100):
    affordance_queries[cls] = top100_affordances[idx]
    affordance_list.append(top100_affordances[idx])
    with torch.no_grad():
        caption_query = [f'an image of {top100_affordances[idx]}']
        caption_targets = clip_tokenizer(caption_query)
        caption_targets = caption_targets.to(device)
        caption_feat = CLIP_Net.encode_text(caption_targets)
        affordance_features[cls] = caption_feat

# get the descriptive queries
descriptive_fp = f'{dataset_root}/metadata/top100_descriptions.txt'
with open(descriptive_fp) as fin:
    top100_descriptions = [line.strip() for line in fin]
descriptive_queries = {}
desciptive_features = {}
descriptive_list = []
for idx, cls in enumerate(top100):
    descriptive_queries[cls] = top100_descriptions[idx]
    descriptive_list.append(top100_descriptions[idx])
    with torch.no_grad():
        caption_query = [f'an image of {top100_descriptions[idx]}']
        caption_targets = clip_tokenizer(caption_query)
        caption_targets = caption_targets.to(device)
        caption_feat = CLIP_Net.encode_text(caption_targets)
        desciptive_features[cls] = caption_feat

# # generate a list of core concepts or train_classes
# # 10 random indices from the top100 classes
# top100_idx = [23, 37, 25, 41, 22, 7, 48, 46, 9, 3]
# longtail_idx = [1233, 1586, 1180, 1047, 1020, 585, 231, 343, 101, 843, 1332, 1050, 1022, 1331, 1188, 228, 1181, 393, 1416, 201, 1178, 525, 709, 745, 798, 496, 1356, 1188, 368, 367, 1313, 885, 1284, 1134, 1597, 496, 1278, 153, 980, 1304, 1346, 850, 969, 978, 1542, 879, 1050, 102, 134, 1273, 736, 629, 1592, 1461, 625, 1462, 1413, 1063, 1105, 978, 838, 853, 1268, 248, 1262, 373, 885, 1368, 757, 747, 1284, 601, 570, 827, 1048, 147, 296, 1512, 873, 1011, 1264, 1224, 1048, 945, 1, 654, 393, 1557, 216, 495, 1110, 3, 562, 721, 1445, 1545, 350, 120, 27, 1447]
# affordance_idx = [12, 3, 32, 37, 8, 11, 14, 18, 2, 31]
# ignore_top_50 = [cls for cls in target_classes if cls not in top50]

# # define the training classes
# train_classes = [top100[idx] for idx in top100_idx]
# # add the longtail classes
# train_classes += [ignore_top_50[idx] for idx in longtail_idx]
# # add values from the affordances
# train_classes += [affordance_list[idx] for idx in affordance_idx]

# get the index of the exclude classes in the top100
exclude_idx = [top100.index(cls) for cls in exclude_classes]
# remove from each list
top100 = [top100[idx] for idx in range(len(top100)) if idx not in exclude_idx]
affordance_list = [affordance_list[idx] for idx in range(len(affordance_list)) if idx not in exclude_idx]
descriptive_list = [descriptive_list[idx] for idx in range(len(descriptive_list)) if idx not in exclude_idx]

# alternatively, we use all the classes
train_classes = top100 + affordance_list + descriptive_list
# train_classes = descriptive_list
print(f'Lenght of train classes: {len(train_classes)}')

# encode the train classes
with torch.no_grad():
    caption_query = [f'an image of {concept}' for concept in train_classes]
    caption_targets = clip_tokenizer(caption_query)
    caption_targets = caption_targets.to(device)
    core_feats = CLIP_Net.encode_text(caption_targets)

mapped_objects = get_mapped_objects(caption_root, dataset_split)

# get all the object feats in this scene
object_feats = torch.stack(mapped_objects['feature'])

# get the similarity between the core classes and the object features
object_feats = object_feats.to(device)
core_feats = core_feats.to(device)
# ensure it sums to 1
object_feats = object_feats / object_feats.norm(dim=-1, keepdim=True)
core_feats = core_feats / core_feats.norm(dim=-1, keepdim=True)
# get the cosine similarity
sim = torch.matmul(core_feats, object_feats.T)

# open the language model
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
pipeline = transformers.pipeline(
"text-generation",
model=model_id,
model_kwargs={"torch_dtype": torch.float16},
device='cuda'
)

# some params for the core class labelling algorithms
acc_thresh = 0.3
# just needs to be large enough to get a concensus early on 
step_size = 50

# iterate through each of the core classes and return the top-k objects
for i in tqdm.tqdm(range(len(core_feats))):
    # # get the top 5 objects
    # topk = torch.topk(sim[i], 1000)
    # # get the topk indices
    # topk_idx = topk.indices
    # # get the topk values
    # topk_val = topk.values

    # check the number of objects with core classes labelled
    labelled_objects = 0
    for k in range(len(mapped_objects['core_classes'])):
        if len(mapped_objects['core_classes'][k]) > 0:
            labelled_objects += len(mapped_objects['core_classes'][k])
    print(f'Number of labelled objects: {labelled_objects}')
    
    # retirm the indices in order of sim value
    topk_val, topk_idx = torch.sort(sim[i], descending=True)
    accuracy_calc = []
    # iterate through each index 
    for j in tqdm(range(len(topk_idx))):
        # get the caption of the selected object
        caption = mapped_objects['descriptions'][topk_idx[j]]
        # print the caption
        print(f'Caption: {caption} Core class: {train_classes[i]}')

        # check it is valid
        if caption == '':
            continue

        # check with LLM that the caption object makes sense
        messages = [
        {"role": "system", "content": "Given two captions, you must decide if they refer to the same object. \
        The input will be two different captions of an object. \
        The output must be in JSON format, do not include other text or explanations. \
        The field ’same_object’ should be set to 'True' if the captions are likely to refer to the same object, and 'False' if not. \
        The field ’explanation’ should describe the reason for this response."},
        {"role": "user", "content": f'caption 1: {caption} \n caption 2: An image of a {train_classes[i]}'},
        ]

        # messages = [
        # {"role": "system", "content": "Given two captions, you must decide if they are likely to belong to refer to the same object. \
        # The input will be two different captions of an object. \
        # The output should 'True' if the captions refer to the same object, and 'False' if not."},
        # {"role": "user", "content": f'caption 1: {caption} \n caption 2: An image of a {train_classes[i]}'},
        # ]

        terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = pipeline(
            messages,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )[0]

        response = outputs['generated_text'][-1]['content']
        # convert to bool
        if 'alse' in response:
            response = False
            accuracy_calc.append(0)
        else:
            response = True
            accuracy_calc.append(1)
            mapped_objects['core_classes'][topk_idx[j]].append(train_classes[i])

        print(f'Response: {response}')

        # check j every 10
        if (j+1) % step_size == 0:
            accuracy = sum(accuracy_calc)/len(accuracy_calc)
            print(f'\nAccuracy: {accuracy}\n')
            if accuracy < acc_thresh:
                break
            accuracy_calc = []


# save the scene objects to file
continual_fp = f'{dataset_root}/finetune_data/continual_top100/{dataset_split}/'

# create this director
os.makedirs(continual_fp, exist_ok=True)

# iterate through each object
for k in range(len(mapped_objects['feature'])):

    # open th h5py file     
    with h5py.File(f'{continual_fp}object_{k}.h5', 'w') as f:
        f.create_dataset('object_feats', data=mapped_objects['feature'][k])
        f.create_dataset('captions', data=mapped_objects['captions'][k])
        f.create_dataset('scene_name', data=mapped_objects['scene_name'][k])
        f.create_dataset('core_classes', data=mapped_objects['core_classes'][k])
        f.create_dataset('descriptions', data=mapped_objects['descriptions'][k])