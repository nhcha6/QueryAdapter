import json
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import transformers
import torch
from huggingface_hub import login
import random
import glob
import h5py
import numpy as np
import clip
import open_clip
import torch.nn as nn
import open3d as o3d
from conceptgraph.utils.eval import compute_pred_gt_associations
from task_definition import *
import pickle

import sys
sys.path.append('../')
sys.path.append('../concept_graphs_data')
from ds import tokenize
from clip_adapter import *
from visualize_cfslam_results import load_result

# add previous directory to path
from clip_adapter import CustomCLIP
from clip_coop import CoOpCLIP

# from utils import load_model

def complete_task_given_captions(task, caption_objects, pcds, affordance_prompt=True):
    # objects_in_scene = {key: caption_objects[key]['gt_class'] for key in caption_objects.keys()}
    # objects_in_scene = json.dumps(objects_in_scene)

    objects_in_scene = [caption_objects[key]['caption'] + ' ' + str(key) for key in caption_objects.keys()]

    # gt_in_scene = {key: caption_objects[key]['gt_class'] for key in caption_objects.keys()}

    input_json = {'objects': objects_in_scene, 'task_definition': task}
    input_json = json.dumps(input_json)

    # baseline message
    if not affordance_prompt:
        messages = [
            {"role": "system", "content": "Your job is to complete tasks in an indoor environment for a user. \
            Given a task and a list of objects, you need to define a feasible plan.\
            Input and output should be in a json file only, do not include any other notes or explainations.\
            The input will be a json dictionary containing a list of 'objects' and a 'task_definition'.\
            The output should be a JSON dictionary defining how the task should be completed.\
            The field 'relevant objects' should contain a list of the objects in the scene that are relevant to the task. \
            The field 'plan' should contain a description of the simplest method to fulfil the task.\
            The field 'objects' should contain a list of the objects that are used in the final plan.\
            Where there are multiple objects of the same class, simply choose one."},
            {"role": "user", "content": input_json},
        ]

    # adding in affordances
    else:
        messages = [
            {"role": "system", "content": "Your job is to complete tasks in an indoor environment for a user. \
            Given a task and a list of objects, you need to define a feasible plan.\
            Input and output should be in a json file only, do not include any other notes or explainations.\
            The input will be a json dictionary containing a list of 'objects' and a 'task_definition'.\
            The output should be a JSON dictionary defining how the task should be completed.\
            The field 'relevant objects' should contain a list of the objects in the scene that are relevant to the task. \
            The field 'plan' should contain a description of the simplest method to fulfil the task.\
            The field 'objects' should contain a list of the objects that are used in the final plan.\
            Where there are multiple objects of the same class, simply choose one.\
            The field 'affordances' should contain a short desciption of the form 'used for <affordance>' to describe the common use case of each object."},
            {"role": "user", "content": input_json},
        ]

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    while True:
        gt_affordance_output = pipeline(
            messages,
            max_new_tokens=2048,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )[0]

        try:
            gt_affordance = gt_affordance_output['generated_text'][-1]['content']
            print(gt_affordance)

            # convert to json
            gt_affordance = json.loads(gt_affordance)

            object_ids = [obj.split(' ')[-1] for obj in gt_affordance['objects']]

            # if the number of object ids is too large, assume the plan is wrong
            if len(object_ids) > 5:
                print("Too many objects")
                # randomly select 5 objects
                object_ids = random.sample(object_ids, 5)

            # get the pcd of the selected object
            pred_pcds = [pcds[int(idx)] for idx in object_ids]

            # make sure the objects are unique
            # pred_objects = list(set(pred_objects))

            # pred_objects = gt_affordance['objects']

            break
        except Exception as e:
            print(e)
            continue

    return pred_pcds

def complete_task_querying_objects(task, object_features, pcds, CLIP_Net, adapter=False, affordance_prompt=True):

    # baseline message
    if not affordance_prompt:
        messages = [
            {"role": "system", "content": "Your job is to complete tasks in an indoor environment for a user. \
            Given a task, you need to define a feasible plan.\
            Output should be in the form of a json file only, do not include any other notes or explainations.\
            The user input will be a definition of a task'.\
            The output should be a JSON dictionary defining how the task should be completed.\
            The field 'referenced_objects' should contain a list of the objects referred to in the task.\
            The field 'plan' should contain a description of the simplest method to fulfil the task.\
            The field 'used_objects' should contain a list of ALL the objects in the environment interacted with in the final plan."},
            {"role": "user", "content": task},
        ]

    else:
        # get the robot to use affordances - use the affordance instead of the referenced objects to get better results?
        # The field 'affordances' should contain a short sentance describing the most common use of each object."},
        
        messages = [
            {"role": "system", "content": "Your job is to complete tasks in an indoor environment for a user. \
            Given a task, you need to define a feasible plan.\
            Output should be in the form of a json file only, do not include any other notes or explainations.\
            The user input will be a definition of a task'.\
            The output should be a JSON dictionary defining how the task should be completed.\
            The field 'referenced_objects' should contain a list of the objects referred to in the task.\
            The field 'plan' should contain a description of the simplest method to fulfil the task.\
            The field 'used_objects' should contain a list of ALL the objects in the environment interacted with in the final plan.\
            The field 'affordances' should contain a short desciption of the form 'used for <affordance>' to describe the common use case of each object."},
            {"role": "user", "content": task},
        ]

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    while True:

        gt_affordance_output = pipeline(
            messages,
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )[0]

        gt_affordance = gt_affordance_output['generated_text'][-1]['content']

        # convert to json
        try:
            gt_affordance = json.loads(gt_affordance)

            print(gt_affordance)

            pred_objects = gt_affordance['used_objects']

            if len(pred_objects) > 5:
                print("Too many objects")
                # randomly select 5 objects
                pred_objects = random.sample(pred_objects, 5)

            # # update the object_query to use a mix of referenced objects and affordances - many different ways to do this
            # # currently use affordances when the object is not in the referenced objects
            # # could use both - an image of a <object> used for <affordance>
            # if affordance_prompt:
            #     # treat referenced and affordance objects differently
            #     referenced_objects = gt_affordance['referenced_objects']
            #     affordances = gt_affordance['affordances']
            #     # generate the object query
            #     object_query = []
            #     for obj in pred_objects:
            #         if obj in referenced_objects:
            #             object_query.append(f'an image of a {obj}')
            #         else:
            #             # get the index of the affordance
            #             # object_query.append(f'an image of an object {affordances[obj]}')
            #             object_query.append(f'an image of a {obj} {affordances[obj]}')
            # # object_query = [f'an image of a {obj} {affordances[obj]}' for obj in pred_objects]
            
            # # otherwise just use the object
            # else:
            
            # encode the query
            object_query = [f'an image of a {obj}' for obj in pred_objects]
            print(object_query)

            # convert to CLIP feature for querying
            query_targets = clip_tokenizer(object_query)
            query_targets = query_targets.to(device)
            query_feat = CLIP_Net.encode_text(query_targets)

            # convert object features to torch
            object_features = torch.tensor(object_features).squeeze(1).to(device)

            if adapter:
                print('Adapter')
                with torch.no_grad():
                    similarities = adapter(object_features, query_feat)
                    # transpose
                    similarities = similarities.T
                    # unsqueeze if necessary
                    if len(similarities.shape) == 1:
                        similarities = similarities.unsqueeze(0)
            else:
                print('Pre-trained')
                # print(query_feat.shape)
                # print(object_features.shape)
                # normalise the features
                query_feat = query_feat / query_feat.norm(dim=-1, keepdim=True)
                object_features = object_features / object_features.norm(dim=-1, keepdim=True)

                # get cosine similarity between query feature and object features
                similarities = torch.matmul(query_feat, object_features.T)
            
            # get the max index
            vals, indices = torch.max(similarities, 1)

            pred_pcds = [pcds[int(idx)] for idx in indices]

            break
        except Exception as e:
            print(e)
            continue

    return pred_pcds

def complete_task_precomputed_queries(task, object_features, pcds, CLIP_Net, adapter=False, affordance_prompt=True):
    pred_objects = task_objects[task]

    # encode the query
    object_query = [f'an image of a {obj}' for obj in pred_objects]
    print(object_query)

    # convert to CLIP feature for querying
    query_targets = clip_tokenizer(object_query)
    query_targets = query_targets.to(device)
    query_feat = CLIP_Net.encode_text(query_targets)

    # convert object features to torch
    object_features = torch.tensor(object_features).squeeze(1).to(device)

    if adapter:
        print('Adapter')
        with torch.no_grad():
            similarities = adapter(object_features, query_feat)
            # transpose
            similarities = similarities.T
            # unsqueeze if necessary
            if len(similarities.shape) == 1:
                similarities = similarities.unsqueeze(0)
    else:
        print('Pre-trained')
        # print(query_feat.shape)
        # print(object_features.shape)
        # normalise the features
        query_feat = query_feat / query_feat.norm(dim=-1, keepdim=True)
        object_features = object_features / object_features.norm(dim=-1, keepdim=True)

        # get cosine similarity between query feature and object features
        similarities = torch.matmul(query_feat, object_features.T)
    
    # get the max index
    vals, indices = torch.max(similarities, 1)

    pred_pcds = [pcds[int(idx)] for idx in indices]

    return pred_pcds

def complete_task_adapter(task, object_features, pcds, CLIP_Net, adapter_features=None, adapter_classes=None):
    pred_objects = task_objects[task]

    # encode the query
    object_query = [f'an image of a {obj}' for obj in pred_objects]

    # convert to CLIP feature for querying
    query_targets = clip_tokenizer(object_query)
    query_targets = query_targets.to(device)
    query_feat = CLIP_Net.encode_text(query_targets)

    # convert object features to torch
    object_features = torch.tensor(object_features).squeeze(1).to(device)

    if adapter_features:
        print('Adapter')
        # figure out which adapter to use
        # get the index of the task in task_sets
        for tasks_idx in range(len(task_sets)):
            if task in task_sets[tasks_idx]:
                break
        
        print(task)
        print(adapter_classes[tasks_idx])

        with torch.no_grad():
            # get the updated query feats
            query_feat = []
            for obj in pred_objects:
                # get the idex of the object in the adapter classes
                idx = adapter_classes[tasks_idx].index(obj)
                query_feat.append(adapter_features[tasks_idx][idx])
            query_feat = torch.stack(query_feat)

            # get similarity between the query and the object features
            object_features = object_features.to(device)
            query_feat = query_feat.to(device)
            # ensure it sums to 1
            object_features = object_features / object_features.norm(dim=-1, keepdim=True)
            query_feat = query_feat / query_feat.norm(dim=-1, keepdim=True)
                        
            # get cosine similarity between query feature and object features
            similarities = torch.matmul(query_feat, object_features.T)

    else:
        print('Pre-trained')
        # print(query_feat.shape)
        # print(object_features.shape)
        # normalise the features
        query_feat = query_feat / query_feat.norm(dim=-1, keepdim=True)
        object_features = object_features / object_features.norm(dim=-1, keepdim=True)

        # get cosine similarity between query feature and object features
        similarities = torch.matmul(query_feat, object_features.T)
    
    # get the max index
    vals, indices = torch.max(similarities, 1)

    pred_pcds = [pcds[int(idx)] for idx in indices]

    return pred_pcds

def get_segment_objects(captions_path):
    scene_objects = {}
    # list all files in the caption root
    instance_files = glob.glob(f'{captions_path}*.pkl')
    # iterate through each file
    for instance_file in tqdm(instance_files):

        scene_name = instance_file.split('/')[-1].split('.')[0]

        # if scene_name not in ['56a0ec536c', '8b5caf3398', '41b00feddb', '98b4ec142f', '7b6477cb95']:
        #     continue

        scene_objects[scene_name] = {
            'features': [],
            'captions': [],
            'pcd': []
        }
        
        with open(instance_file, 'rb') as fin:
            segments = pickle.load(fin)

            # update the pcds
            for idx in range(len(segments['object_feats'])):
                pcd = segments['object_pcd'][idx]
                xfI = segments['object_feats'][idx]

                scene_objects[scene_name]['features'].append(xfI)
                scene_objects[scene_name]['captions'].append('object')
                scene_objects[scene_name]['pcd'].append(pcd)

    return scene_objects

def get_conceptgraph_objects(captions_path):
    scene_objects = {}
    # list all files in the caption root
    instance_files = glob.glob(f'{captions_path}*.h5')
    # iterate through each file
    for instance_file in tqdm(instance_files):

        scene_name = instance_file.split('/')[-1].split('_')[0]

        # if scene_name not in ['56a0ec536c', '8b5caf3398', '41b00feddb', '98b4ec142f', '7b6477cb95']:
        #     continue
        
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
                # captions = [f'an image of a {gt_class}']
                pcd = f['pcd'][()]

                # get the caption data
                # init_captions = f['captions']
                object_tag = f['object_tags'][()].decode('utf-8')
                object_description = f['object_description'][()].decode('utf-8')
                object_affordance = f['object_affordance'][()].decode('utf-8')
            except Exception as e:
                print(e)
                continue

        # check if scene_name is in the scenes
        if scene_name not in scene_objects.keys():
            # dictionary to store the object features
            scene_objects[scene_name] = {
                'features': [],
                'captions': [],
                'pcd': []
            }
            scene_objects[scene_name]['features'] = [xfI]
            scene_objects[scene_name]['captions'] = [object_tag]
            scene_objects[scene_name]['pcd'] = [pcd]
        else:
            scene_objects[scene_name]['features'].append(xfI)
            scene_objects[scene_name]['captions'].append(object_tag)
            scene_objects[scene_name]['pcd'].append(pcd)
    return scene_objects

def get_gt_boxes(image_root, scene_name, target_classes):

    gt_seg_path = str(image_root)+f'/data/{scene_name}/scans/segments_anno.json'
    gt_pcd_path = str(image_root)+f'/data/{scene_name}/scans/mesh_aligned_0.05_semantic.ply'
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

def evaluate_task_queries(method='adapter', affordance_prompt=True):
    count = 0
    results = {'TP': 0, 'FP': 0, 'FN': 0}

    if 'segments' in method:
        captions_path = f'{data_source}/finetune_data/preprocessed_segments/val/'
        scene_objects = get_segment_objects(captions_path)
    else:
        captions_path = f'{data_source}/finetune_data/conceptgraph/val/'
        scene_objects = get_conceptgraph_objects(captions_path)

    # get the adapter if needed
    if 'adapter' in method:
        # path to the adapters
        # adapter_path = '/home/nicolas/hpc-home/ProbVLM/results_task/1_TaskAdapter_Medium/output/'
        adapter_path = '/home/nicolas/hpc-home/ProbVLM/final_results/task_tests/0_TaskAdapter_Tasks/output/'
        model = 'epoch50'
        # adapter features and objects
        adapter_features = [None for _ in range(10)]
        adapter_classes = [None for _ in range(10)]
        # iterate through each folder in the adapter path
        for folder in os.listdir(adapter_path):
            # extract the final number in the name
            folder_number = int(folder.split('_')[-1])
            # get the tasks related to this adapter
            tasks = task_sets[folder_number]
            # get the objects in this task
            objects = []
            for task in tasks:
                objects.extend(task_objects[task])
            # get the unique objects
            all_classes = list(set(objects))

            adapter_cfg = mch({
                'classnames': all_classes,
                'embedding_dim': 1024,
                'logit_scale': np.log(50),
            })
            adapter = CoOpCLIP(adapter_cfg, CLIP_Net, device, torch.float32, clip_tokenizer)
            # adapter = CustomCLIP(adapter_cfg, CLIP_Net, device)
            checkpoint = torch.load(f'{adapter_path}{folder}/{model}.pth', weights_only=True)
            # print(checkpoint)
            # print(adapter.adapter.ctx)
            adapter.adapter.ctx = checkpoint
            
            # get the text features
            with torch.no_grad():
                coop_text_feats = adapter.return_clip_feats()
                print(coop_text_feats.shape)
                # add the features to the adapter features
                adapter_features[folder_number]=coop_text_feats
                adapter_classes[folder_number]=all_classes

    for scene in scene_objects.keys():
        # print(scene)
        # if scene != '3f15a9266d':
        #     continue

        # make sure the scene has been processed by conceptgraphs
        try:
            # open the pcd file generate using conceptgraph
            pcd_path = f'{data_source}/conceptgraph_data/{scene}/pcd_saves/full_pcd_none_overlap_maskconf0.95_simsum1.2_dbscan.1_merge20_masksub_post.pkl.gz'
            # get the objects
            fg_objects, bg_objects, class_colors = load_result(pcd_path)
            # print(f'Loaded conceptgraph with {len(fg_objects)} objects')
        except Exception as e:
            # print(e)
            continue

        # get a list of the objects
        semantic_data = f'{data_source}/data/{scene}/scans/segments_anno.json'
        with open(semantic_data) as f:
            data = json.load(f)

        objects_in_scene = []
        for i, instance in enumerate(data['segGroups']):
            exclude_classes = ['ceiling', 'floor', 'wall']
            if instance['label'] in exclude_classes:
                continue
            objects_in_scene.append(instance['label'])

        # get the gt objects
        gt_boxes, gt_labels, gt_pcd, pcd_disp, gt_annos = get_gt_boxes(data_source, scene, semantic_classes)

        # stack the objects
        object_features = np.stack(scene_objects[scene]['features'], axis=0)
        caption_objects = {i: {'caption': scene_objects[scene]['captions'][i], 'gt_class': scene_objects[scene]['captions'][i]} for i in range(len(scene_objects[scene]['captions']))}
        object_pcds = scene_objects[scene]['pcd']

        # print(object_features.shape)
        # print(caption_objects)
        # print(objects_in_scene)

        # iterate through the tasks
        for task in task_list.keys():

            relevant_objects = task_list[task]

            skip_task = False
            # at least one object in each set should be present to complete the task
            for object_set in relevant_objects:
                object_set_present = False
                for obj in object_set:
                    if obj in objects_in_scene:
                        # skipping this task
                        object_set_present = True
                if not object_set_present:
                    skip_task = True
                    break
            if skip_task:
                continue
                
            count+=1

            # print the task
            print(f"\nTask {count}: {task}")
            print(f"GT Objects: {relevant_objects}")

            # get the pred objects
            if 'pretrained' in method:
                # pred_pcds = complete_task_querying_objects(task, object_features, object_pcds, CLIP_Net, adapter=False, affordance_prompt=affordance_prompt)
                pred_pcds = complete_task_precomputed_queries(task, object_features, object_pcds, CLIP_Net, adapter=False, affordance_prompt=affordance_prompt)
            elif 'adapter' in method:
                pred_pcds = complete_task_adapter(task, object_features, object_pcds, CLIP_Net, adapter_features=adapter_features, adapter_classes=adapter_classes)
            else:
                pred_pcds = complete_task_given_captions(task, caption_objects, object_pcds, affordance_prompt=affordance_prompt)

            pred_objects = []
            for pred in pred_pcds:
                # match the pred_pcds to the object features
                pred_gt_associations = compute_pred_gt_associations(torch.tensor(pred).unsqueeze(0).cuda().contiguous().float(), torch.tensor(gt_pcd).unsqueeze(0).cuda().contiguous().float()) 
                # print(pred_gt_associations[0].shape)
                # get the annos
                pred_annos = gt_annos[pred_gt_associations[0].detach().cpu().numpy()]
                unique, counts = np.unique(pred_annos, return_counts=True)
                pred_annos = [semantic_classes[anno] for anno in pred_annos  if anno<len(semantic_classes)]
                # get a count of each unique element in pred_annos
                unique, counts = np.unique(pred_annos, return_counts=True)
                # select the class with from nuique the maximum count
                try:
                    gt_class = unique[np.argmax(counts)]
                    pred_objects.append(gt_class)
                except:
                    pred_objects.append('None')
                    continue
                

            # get the results - we need at least one object from each set
            correct_objects = 0 # TP
            missing_objects = 0 # FN
            for object_set in relevant_objects:
                set_predicted = False
                for obj in object_set:
                    if obj in pred_objects:
                        # skipping this task
                        set_predicted = True
                if set_predicted:
                    correct_objects += 1
                else:
                    missing_objects += 1
            
            # now check for false positives
            extra_objects = 0 # FP
            task_object_set = [obj for object_set in relevant_objects for obj in object_set]
            for obj in pred_objects:
                if obj not in task_object_set:
                    extra_objects += 1
        
            # # calculate the number of predicted objects that are correct using set intersection
            # correct_objects = len(set(task_objects).intersection(set(pred_objects))) # TP
            # # calculate the number of objects that are not in the prediction
            # missing_objects = len(set(task_objects).difference(set(pred_objects))) # FN
            # # calculate the number of objects that are in the prediction but not in the task
            # extra_objects = len(set(pred_objects).difference(set(task_objects))) # FP

            # update the results
            results['TP'] += correct_objects
            results['FN'] += missing_objects
            results['FP'] += extra_objects

            print(results)
    
    return results

data_source = '../../../Datasets/scannetpp/data_download/complete_dataset'

# potential object that could be added in these tasks
# recycling bin

# open the val splits
splits_path = f'{data_source}/splits/nvs_sem_val.txt'
# read each line and add to the splits file
with open(splits_path) as fin:
    scenes = [line.strip() for line in fin]

# get the semantic classes
semantic_classes = f'{data_source}/metadata/semantic_classes.txt'
with open(semantic_classes) as fin:
    semantic_classes = [line.strip() for line in fin]
# remove the exclude classes
exclude_classes = ['ceiling', 'floor', 'wall', 'object']
semantic_classes = [cls for cls in semantic_classes if cls not in exclude_classes]

# print(semantic_classes)

# open the language model
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.float16},
    device='cuda'
)

# CLIP_Net = load_model(device='cuda', model_path=None)
device = 'cuda'
CLIP_Net, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-H-14", "laion2b_s32b_b79k"
)
CLIP_Net = CLIP_Net.to(device)
CLIP_Net.eval()
clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

# checkpoint_path = '../ckpt/08_Caption_Grid_Search/ratio0.3_thresh0.24/best.pth'

# adapter_cfg = mch({
#     'ratio': 0.3,
#     'mean_score': 0.2,
#     'logit_scale': torch.log(torch.tensor(50.))
# })

# adapter = CustomCLIP(adapter_cfg, CLIP_Net, device)
# checkpoint = torch.load(checkpoint_path, weights_only=True)
# adapter.load_state_dict(checkpoint)

# choose which prompting style
affordance_prompt = False

# reduce the number of tasks for testing
# scenes = scenes[:2]

method_comparison = {}
# for method in ['adapter_segments', 'pretrained_segments', 'adapter_conceptgraph', 'pretrained', 'caption']:
for method in ['adapter_segments', 'adapter_conceptgraph']:
    results = evaluate_task_queries(method=method, affordance_prompt=affordance_prompt)  
    method_comparison[method] = results

# print the results
print("Affordance prompt: ", affordance_prompt)
for key, val in method_comparison.items():
    print(f"\n{key}: {val}")