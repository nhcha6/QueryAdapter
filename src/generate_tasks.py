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

# import sys
# sys.path.append('../')
from ds import tokenize
from clip_adapter import *
from utils import load_model

def complete_task_given_captions(task, caption_objects, use_gt=False, affordance_prompt=True):
    # objects_in_scene = {key: caption_objects[key]['gt_class'] for key in caption_objects.keys()}
    # objects_in_scene = json.dumps(objects_in_scene)

    if use_gt:
        print('Using GT')
        objects_in_scene = [caption_objects[key]['gt_class'] + ' ' + str(key) for key in caption_objects.keys()]
    else:
        print('Using captions')
        objects_in_scene = [caption_objects[key]['caption'] + ' ' + str(key) for key in caption_objects.keys()]

    gt_in_scene = {key: caption_objects[key]['gt_class'] for key in caption_objects.keys()}

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

            # get the gt of the predicted objects
            pred_objects = [gt_in_scene[int(idx)] for idx in object_ids]

            # make sure the objects are unique
            pred_objects = list(set(pred_objects))

            # pred_objects = gt_affordance['objects']

            break
        except Exception as e:
            print(e)
            continue

    print(f"With objects: {pred_objects}")

    return pred_objects

def complete_task_querying_objects(task, object_features, caption_objects, CLIP_Net, adapter=False, affordance_prompt=True):

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

            # update the object_query to use a mix of referenced objects and affordances - many different ways to do this
            # currently use affordances when the object is not in the referenced objects
            # could use both - an image of a <object> used for <affordance>
            if affordance_prompt:
                # treat referenced and affordance objects differently
                referenced_objects = gt_affordance['referenced_objects']
                affordances = gt_affordance['affordances']
                # generate the object query
                object_query = []
                for obj in pred_objects:
                    if obj in referenced_objects:
                        object_query.append(f'an image of a {obj}')
                    else:
                        # get the index of the affordance
                        # object_query.append(f'an image of an object {affordances[obj]}')
                        object_query.append(f'an image of a {obj} {affordances[obj]}')
            # object_query = [f'an image of a {obj} {affordances[obj]}' for obj in pred_objects]
            
            # otherwise just use the object
            else:
                object_query = [f'an image of a {obj}' for obj in pred_objects]
            print(object_query)

            # convert to CLIP feature for querying
            query_targets = target_transform(object_query)
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

            # get the pred object classes
            pred_objects = [caption_objects[idx.item()]['gt_class'] for idx in indices]
            pred_objects = list(set(pred_objects))

            print(f"Without objects: {pred_objects}")
            break
        except Exception as e:
            print(e)
            continue

    return pred_objects


def evaluate_task_queries(method='adapter', affordance_prompt=True):
    count = 0
    results = {'TP': 0, 'FP': 0, 'FN': 0}

    scene_objects = {}
    captions_path = f'{data_source}/finetune_data/conceptgraphs/val/'
    # list all files in the caption root
    instance_files = glob.glob(f'{captions_path}*.h5')
    print(instance_files)
    # iterate through each file
    for instance_file in tqdm(instance_files):
        
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

        # check if scene_name is in the scenes
        if scene_name not in scene_objects.keys():
            scene_objects[scene_name]['features'] = [xfI]
            scene_objects[scene_name]['captions'] = [caption]
        else:
            scene_objects[scene_name]['features'].append(xfI)
            scene_objects[scene_name]['captions'].append(caption)

    for scene in scene_objects.keys():
        # print(scene)
        # if scene != '3f15a9266d':
        #     continue

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
        # print(len(objects_in_scene))



        # # get the corresponding captions
        # captions_path = f'{data_source}/finetune_data/dataset01/val_caption/{scene}'
        # # use glob to get all files with the scene name
        # caption_files = glob.glob(captions_path+'*')

        # # iterate through the files and get the captions
        # caption_objects = {}
        # objects_in_scene = []
        # object_features = []
        # for i, file in enumerate(caption_files):
        #     with h5py.File(file, 'r') as f:
        #         object_tag = f['object_tag'][()].decode('utf-8').replace('_', ' ')
        #         # object_description = f['object_description'][()].decode('utf-8').replace('_', ' ')
        #         # object_affordance = f['object_affordance'][()].decode('utf-8').replace('_', ' ')
        #     # replace val_caption with val in the file
        #     with h5py.File(file.replace('val_caption','val'), 'r') as f:
        #         class_nums = f['class_nums']
        #         cls_num = class_nums[0]

        #         # get the visual features
        #         visual_features = f['visual_features']
        #         xfI = np.mean(visual_features, axis=0)
        
        #     object_features.append(xfI)
        #     caption_objects[i] = {'caption': object_tag, 'gt_class': semantic_classes[cls_num]}
        #     objects_in_scene.append(semantic_classes[cls_num])
        # # print(len(objects_in_scene))

        # stack the objects
        object_features = np.stack(scene_objects['features'], axis=0)
        caption_objects = {i: {'caption': scene_objects['captions'][i], 'gt_class': scene_objects['captions'][i]} for i in range(len(objects_in_scene))}

        print(object_features.shape)
        print(caption_objects)
        print(objects_in_scene)

        # iterate through the tasks
        for task in task_list.keys():

            task_objects = task_list[task]

            skip_task = False
            # at least one object in each set should be present to complete the task
            for object_set in task_objects:
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
            print(f"GT Objects: {task_objects}")

            # without giving the objects
            if method == 'adapter':
                pred_objects = complete_task_querying_objects(task, object_features, caption_objects, CLIP_Net, adapter=adapter, affordance_prompt=affordance_prompt)
            elif method == 'pretrained':
                pred_objects = complete_task_querying_objects(task, object_features, caption_objects, CLIP_Net, adapter=False, affordance_prompt=affordance_prompt)
            elif method == 'caption':
                pred_objects = complete_task_given_captions(task, caption_objects, use_gt=False, affordance_prompt=affordance_prompt)
            elif method == 'gt':
                pred_objects = complete_task_given_captions(task, caption_objects, use_gt=True, affordance_prompt=affordance_prompt)

            # get the results - we need at least one object from each set
            correct_objects = 0 # TP
            missing_objects = 0 # FN
            for object_set in task_objects:
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
            task_object_set = [obj for object_set in task_objects for obj in object_set]
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

data_source = '../../Datasets/scannetpp/data_download/complete_dataset'

# potential object that could be added in these tasks
# recycling bin

# list the tasks
task_list = {"clean the cup in the sink and put it in the cabinet": [['sink'], ['cabinet', 'kitchen cabinet'], ['cup']], 
             "clean the plate in the sink and put it in the cabinet": [['sink'], ['cabinet', 'kitchen cabinet'], ['plate']], 
             "clean the bowl in the sink and put it in the cabinet": [['sink'], ['cabinet', 'kitchen cabinet'], ['bowl']],
             "clean the pot in the sink and put it in the cabinet": [['sink'], ['cabinet', 'kitchen cabinet'], ['pot']],
             "clean the pan in the sink and put it in the cabinet": [['sink'], ['cabinet', 'kitchen cabinet'], ['pan']],
             "clean the mug in the sink and put it in the cabinet": [['sink'], ['cabinet', 'kitchen cabinet'], ['mug']],
            "clean the bottle in the sink and put it in the cabinet": [['sink'], ['cabinet', 'kitchen cabinet'], ['bottle']],
            "put the book away": [['book', 'books'], ['bookshelf', 'book shelf', 'shelf']],
            "put the shoes away": [['shoe', 'shoes'], ['shoe rack']],
            "clean the writing off the whiteboard": [['whiteboard'], ['whiteboard eraser']],
            "draw a picture": [['paper'], ['pen']],
            "water the plant with the bucket": [['plant', 'plant pot', 'potted plant'], ['bucket']],
            "water the plant with the bottle": [['plant', 'plant pot', 'potted plant'], ['bottle']],
            "let some light in from outside": [['window', 'window sill'], ['blind', 'blinds','window blind', 'curtain']],
            'get me a cup of tap water': [['cup'], ['sink', 'tap']],
            'get me a bottle of tap water': ['bottle', ['sink', 'tap']],
            'use my laptop to play some music out loud': [['laptop'], ['speaker', 'headphones']],
            'make sure someone can sit at at my desk': [['chair', 'office chair'], ['desk']],
            'make sure someone can sit at the table': [['chair', 'dining chair'], ['table']],
            'bring me something disposable to dry my hands, then throw it away': [['paper towel'], ['trash can']],
            'bring me something disposable to clean the table, then throw it away': [['paper towel'], ['trash can']],
            'put a chair somewhere warm for me to sit': [['chair', 'office chair', 'dining chair'], ['heater', 'window', 'window sill']],
            'the plant is not getting enough light, move it to a better spot': [['plant', 'plant pot', 'potted plant'], ['window', 'window sill']],
            'turn on the TV and make sure it is not too bright': [['tv'], ['blind', 'blinds', 'window blind', 'light switch', 'ceiling lamp', 'ceiling light', 'table lamp', 'floor lamp']],
            'find me a book and make sure it is bright enough to read': [['book', 'books', 'bookshelf', 'book shelf'], ['blind', 'blinds', 'window blind', 'light switch', 'ceiling lamp', 'ceiling light', 'table lamp', 'floor lamp']],
            'dispose of this box for me': [['box', 'crate', 'cardboard box'], ['trash can']],
            'throw away this paper': [['paper'], ['trash can']],
            'relocate the pillows so they are ready for bed time': [['pillow', 'pillows', 'cushion'], ['bed']],
             }



# open the val splits
splits_path = f'{data_source}/splits/nvs_sem_val.txt'
# read each line and add to the splits file
with open(splits_path) as fin:
    scenes = [line.strip() for line in fin]

# get the semantic classes
semantic_classes = f'{data_source}/metadata/semantic_classes.txt'
with open(semantic_classes) as fin:
    semantic_classes = [line.strip() for line in fin]
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
for method in ['pretrained', 'caption']:
# for method in ['adapter', 'pretrained']:
    results = evaluate_task_queries(method=method, affordance_prompt=affordance_prompt)  
    method_comparison[method] = results

# print the results
print("Affordance prompt: ", affordance_prompt)
for key, val in method_comparison.items():
    print(f"\n{key}: {val}")