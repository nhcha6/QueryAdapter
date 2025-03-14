import os
import glob
import tqdm
import h5py
import torch
import numpy as np
import cv2
from PIL import Image
from visualize_cfslam_results import load_result
import open_clip
import json
import open3d as o3d
from conceptgraph.utils.ious import *
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import transformers
from huggingface_hub import login
import threading
import gc
from conceptgraph.utils.eval import compute_pred_gt_associations

# add higgingface hub login token
# login(<token>)

import sys
sys.path.append('../')
from ds._dataloader import tokenize, imagenet_transform_fn

def clip_cropped_images(instance, clip_model, clip_preprocess, padding=20):
    # iterate through each detection to get the data
    visual_features = []
    cropped_images = []
    for i in range(len(instance['xyxy'])):
        # Get the crop of the mask with padding
        x_min, y_min, x_max, y_max = instance['xyxy'][i]

        # get the image
        image = Image.open(instance['color_path'][i])

        # Check and adjust padding to avoid going beyond the image borders
        image_width, image_height = image.size
        left_padding = min(padding, x_min)
        top_padding = min(padding, y_min)
        right_padding = min(padding, image_width - x_max)
        bottom_padding = min(padding, image_height - y_max)

        # Apply the adjusted padding
        x_min -= left_padding
        y_min -= top_padding
        x_max += right_padding
        y_max += bottom_padding

        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        cropped_images.append(cropped_image)

        # detected_boxes.append(instance['bbox'])
        # gt_boxes.append(instance['bbox'])
        
        # Get the preprocessed image for clip from the crop 
        with torch.no_grad():
            preprocessed_image = clip_preprocess(cropped_image).unsqueeze(0).to("cuda")

            crop_feat = clip_model.encode_image(preprocessed_image)
            crop_feat /= crop_feat.norm(dim=-1, keepdim=True)

        # list of clip feats
        visual_features.append(crop_feat)

    # stack the visual features
    visual_features = torch.stack(visual_features)
    # convert to numpy
    visual_features = visual_features.cpu().numpy()

    return visual_features, cropped_images

def aligned_box_from_pcd(obj_pcd):
    obj_pcd = np.asarray(obj_pcd.points)
    obb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(obj_pcd))
    obb = [obb.get_box_points()]
    obb = np.asarray(obb)
    obb = torch.from_numpy(obb).to('cuda')
    return obb

def find_gt_object(instance, gt_boxes, gt_labels):
    # get the axis aligned bbox of the object
    obj_pcd = instance['pcd']
    obb = aligned_box_from_pcd(obj_pcd)

    # get the spatial similarity
    spatial_sim = compute_iou_batch(obb, gt_boxes)
    # get the maximum value and index
    max_val, max_idx = spatial_sim.max(dim=1)
    # get the label
    label = gt_labels[max_idx]

    return max_val.item(), label

def get_gt_boxes(scene, conceptgraph_root, target_classes):
    # gt segmentation path
    gt_seg_path = f'{conceptgraph_root}/data/{scene}/scans/segments_anno.json'
    gt_pcd_path = f'{conceptgraph_root}/data/{scene}/scans/mesh_aligned_0.05_semantic.ply'
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

def match_pred_to_gt(pcd, gt_pcd, gt_annos):
    # match the object to the gt
    pred_gt_associations = compute_pred_gt_associations(torch.tensor(pcd).unsqueeze(0).cuda().contiguous().float(), torch.tensor(gt_pcd).unsqueeze(0).cuda().contiguous().float()) 
    
    pred_annos = gt_annos[pred_gt_associations[0].detach().cpu().numpy()]
    unique, counts = np.unique(pred_annos, return_counts=True)
    pred_annos = [semantic_classes[anno] for anno in pred_annos if anno<len(semantic_classes)]
    # get a count of each unique element in pred_annos
    unique, counts = np.unique(pred_annos, return_counts=True)

    # get the ratio of the max value to the total
    ratios =[val/pred_gt_associations[0].shape[0] for val in counts]
    # get indices where ratio is above a threshold
    above_thresh = [i for i in range(len(ratios)) if ratios[i] > 0.3]
    # get the labels of the above threshold
    gt_labels = [unique[i] for i in above_thresh if unique[i] in semantic_classes]

    return gt_labels

def process_scene(scene): 
    # check if the scene has already been processed by checking if folder exists
    if os.path.exists(f'{caption_root}/{split}_1/{scene}_0.h5'):
        print(f'Scene {scene} already processed')
        return

    # make sure the scene has been processed by conceptgraphs
    try:
        # open the pcd file generate using conceptgraph
        pcd_path = f'{conceptgraph_root}/conceptgraph_data/{scene}/pcd_saves/full_pcd_none_overlap_maskconf0.95_simsum1.2_dbscan.1_merge20_masksub_post.pkl.gz'
        # get the objects
        objects, bg_objects, class_colors = load_result(pcd_path)
    except Exception as e:
        print(e)
        return

    # generate an instance dictionary
    instance_data = {}
    # padding
    padding = 20  # Adjust the padding amount as needed
    
    # get gt boxes
    # gt_boxes, gt_labels = get_gt_boxes(scene, conceptgraph_root)

    # Initialize the CLIP model
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", "laion2b_s32b_b79k"
    )
    clip_model = clip_model.to('cuda')
    clip_model.eval()
    clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

    # get gt information for the scene
    gt_boxes, gt_labels, gt_pcd, pcd_disp, gt_annos = get_gt_boxes(scene, conceptgraph_root, semantic_classes)

    # iterate through each instance
    for idx, instance in enumerate(objects):
        # iterate through each key in the instance
        # for key in instance.keys():
        #     print(key)

        visual_features, cropped_images = clip_cropped_images(instance, clip_model, clip_preprocess, padding)

        # get the pcd
        pcd = np.array(instance['pcd'].points)

        # max_val, label = find_gt_object(instance, gt_boxes, gt_labels)

        # get the annos
        matched_labels = match_pred_to_gt(pcd, gt_pcd, gt_annos)

        # # display the cropped image
        # print(label, max_val)
        # disp = np.array(cropped_images[0])
        # import matplotlib.pyplot as plt
        # plt.imshow(disp)
        # plt.show()

        instance_data[idx] = {}
        # append the data
        instance_data[idx]['image_names'] = instance['color_path']
        instance_data[idx]['crops'] = instance['xyxy']
        instance_data[idx]['visual_features'] = visual_features
        instance_data[idx]['object_feat'] = instance['clip_ft']
        instance_data[idx]['pcd'] = np.array(instance['pcd'].points)
        instance_data[idx]['cropped_images'] = cropped_images
        instance_data[idx]['gt_class'] = matched_labels

    # close the clip model
    del clip_model
    del clip_preprocess
    del clip_tokenizer
    
    # prepare the image caption system
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
    model.to("cuda:0")
    prompt = "USER: <image>\nDescribe the central object in the image in one sentance. ASSISTANT:"

    # iterate through each instance in the scene
    object_captions = {}
    for idx in instance_data.keys():
        try:
            # select the images for processing
            cropped_images = instance_data[idx]['cropped_images']
            # select the 10 largest crops if more than 10
            if len(cropped_images) > 10:
                sizes = [crop.size[0]*crop.size[1] for crop in cropped_images]
                largest_crops = np.argsort(sizes)[::-1][:10]
                cropped_images = [cropped_images[i] for i in largest_crops]

            object_captions[idx] = {}
            captions = []
            for img in cropped_images:
                inputs = processor(images=img, text=prompt, return_tensors="pt").to("cuda:0")
                # Generate
                generate_ids = model.generate(**inputs, max_new_tokens=60)
                output = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]

                cap = output.split('ASSISTANT: ')[1]
                captions.append(cap)
            object_captions[idx]['captions'] = captions
        except Exception as e:
            print(e)
            continue

    # del the captioner
    del model
    del processor

    # open the language model
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.float16},
    device_map="cuda:0",
    )

    # iterate through each instance in the scene
    for idx in object_captions.keys():
        try:
            # convert the captions to a json file
            json_caption = {'captions': object_captions[idx]['captions']}
            captions = json.dumps(json_caption)

            messages = [
                {"role": "system", "content": "Identify and describe objects in scenes. Input and output must be in JSON format, do not include any other text or explainations. \
                The input field ’captions’ contains a list of image captions aiming to identify the object. \
                An object mentioned multiple times is likely accurate. \
                If various objects are repeated and a container/surface is noted such as a shelf or table, assume the (repeated) objects are on that container/surface. \
                The field ’possible object_tags’ should list potential object categories. \
                The field ’object tag’ should contain the central object addressed by the captions. \
                The field 'affordance' should contain a short desciption of the form 'used for <affordance>' to describe the common use case of the object. \
                The field 'object description' should contain a one sentance description of the appearance of the central object. \
                For unrelated, non-repeating (or empty) captions set ’object tag’ to ’invalid’. \
                Focus on indoor object types, as the input captions are from indoor scans."},
                {"role": "user", "content": captions},
            ]

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

            response = json.loads(outputs['generated_text'][-1]['content'])
            # convert response string to a json
            object_captions[idx]['object_tag'] = response['object_tag']
            object_captions[idx]['object_description'] = response['object_description']
            object_captions[idx]['object_affordance'] = response['affordance']
        except Exception as e:
            print(e)
            continue

    del pipeline

    # then, we convert it into a dataset for the sampler
    for idx in object_captions.keys():
        try:
            instance_data[idx]['image_names']
            instance_data[idx]['crops']
            instance_data[idx]['visual_features']
            instance_data[idx]['object_feat']
            instance_data[idx]['pcd']
            instance_data[idx]['gt_class']

            object_captions[idx]['object_tag']
            object_captions[idx]['object_description']
            object_captions[idx]['object_affordance']
            object_captions[idx]['captions']

            # open th h5py file
            with h5py.File(f'{caption_root}/{split}_1/{scene}_{idx}.h5', 'w') as f:

                # raw concepgraphs data
                f.create_dataset('image_names', data=instance_data[idx]['image_names'])
                f.create_dataset('crops', data=instance_data[idx]['crops'])
                f.create_dataset('visual_features', data=instance_data[idx]['visual_features'])
                f.create_dataset('object_feat', data=instance_data[idx]['object_feat'])
                f.create_dataset('pcd', data=instance_data[idx]['pcd'])
                f.create_dataset('gt_class', data=instance_data[idx]['gt_class'])

                # caption data
                f.create_dataset('object_tags', data=object_captions[idx]['object_tag'])
                f.create_dataset('object_description', data=object_captions[idx]['object_description'])
                f.create_dataset('object_affordance', data=object_captions[idx]['object_affordance'])
                f.create_dataset('captions', data=object_captions[idx]['captions'])

                # # raw conceptgraphs data
                # f.create_dataset('image_names', data=instance_data[idx]['image_names'])
                # f.create_dataset('crops', data=instance_data[idx]['crops'])
                # # f.create_dataset('captions', data=instance_data[instance]['captions'])
                # # f.create_dataset('class_nums', data=instance_data[instance]['class_nums'])
                # f.create_dataset('visual_features', data=instance_data[idx]['visual_features'])
                # # f.create_dataset('text_features', data=instance_data[instance]['text_features'])
                # f.create_dataset('object_feat', data=instance_data[idx]['object_feat'])
                # f.create_dataset('pcd', data=instance_data[idx]['pcd'])
                # f.create_dataset('gt_class', data=instance_data[idx]['gt_class'])
                
                # # caption data
                # f.create_dataset('object_tags', data=object_captions[idx]['object_tag'])
                # f.create_dataset('object_description', data=object_captions[idx]['object_description'])
                # f.create_dataset('object_affordance', data=object_captions[idx]['object_affordance'])
                # f.create_dataset('captions', data=object_captions[idx]['captions'])
        except Exception as e:
            print(e)
            continue

# define the data root
conceptgraph_root = '/home/nicolas/hpc-home/Datasets/scannetpp/data_download/complete_dataset'
caption_root = '/home/nicolas/hpc-home/Datasets/scannetpp/data_download/complete_dataset/finetune_data/conceptgraph'
split = 'val'

# get the dataset splits
# splits_path = image_root+f'/data_download/complete_dataset/splits/nvs_sem_{split}.txt'
splits_path = f'{conceptgraph_root}/splits/nvs_sem_{split}.txt'
# read each line and add to the splits file
with open(splits_path) as fin:
    scenes = [line.strip() for line in fin]

# open the semantic classes
semantic_classes = []
with open(f'{conceptgraph_root}/metadata/semantic_classes.txt') as fin:
    for line in fin:
        semantic_classes.append(line.strip())

# create the caption root directory
os.makedirs(f'{caption_root}/{split}', exist_ok=True)

scene_n = 0
for scene in tqdm.tqdm(scenes):

    # # start thread to ensure memory is released 
    # thread = threading.Thread(target=process_scene, args=(scene,))

    # # start the thread
    # thread.start()

    # # wait for the thread to finish
    # thread.join()

    # process the scene
    process_scene(scene)

    gc.collect()


