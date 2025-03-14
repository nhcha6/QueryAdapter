'''
The script is used to model Grounded SAM detections in 3D, it assumes the tag2text classes are avaialable. It also assumes the dataset has Clip features saved for each object/mask.
'''

# Standard library imports
import copy
from datetime import datetime
import os
from pathlib import Path
import gzip
import pickle
import json

# Related third party imports
from PIL import Image
import cv2
import imageio
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
import open_clip
from visualize_cfslam_results import load_result
from munch import Munch as mch

import hydra
import omegaconf
from omegaconf import DictConfig
import h5py

# Local application/library specific imports
from conceptgraph.dataset.datasets_common import get_dataset
from conceptgraph.utils.vis import OnlineObjectRenderer
from conceptgraph.utils.ious import (
    compute_2d_box_contained_batch,
    compute_3d_iou_accuracte_batch
)
from conceptgraph.utils.general_utils import to_tensor

from conceptgraph.slam.slam_classes import MapObjectList, DetectionList
from conceptgraph.slam.utils import (
    create_or_load_colors,
    merge_obj2_into_obj1, 
    denoise_objects,
    filter_objects,
    merge_objects, 
    gobs_to_detection_list,
    process_pcd,
)
from conceptgraph.slam.mapping import (
    compute_spatial_similarities,
    compute_visual_similarities,
    aggregate_similarities,
    merge_detections_to_objects
)
from scannetpp_dataset import ScannetppDataset, get_scannet_dataset
from conceptgraph.utils.eval import compute_pred_gt_associations

# add previous directory to path
import sys
sys.path.append('../')
from clip_adapter import CustomCLIP
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import transformers


BG_CLASSES = ["wall", "floor", "ceiling"]

# Disable torch gradient computation
torch.set_grad_enabled(False)

def compute_match_batch(cfg, spatial_sim: torch.Tensor, visual_sim: torch.Tensor) -> torch.Tensor:
    '''
    Compute object association based on spatial and visual similarities
    
    Args:
        spatial_sim: a MxN tensor of spatial similarities
        visual_sim: a MxN tensor of visual similarities
    Returns:
        A MxN tensor of binary values, indicating whether a detection is associate with an object. 
        Each row has at most one 1, indicating one detection can be associated with at most one existing object.
        One existing object can receive multiple new detections
    '''
    assign_mat = torch.zeros_like(spatial_sim)
    if cfg.match_method == "sim_sum":
        sims = (1 + cfg.phys_bias) * spatial_sim + (1 - cfg.phys_bias) * visual_sim # (M, N)
        row_max, row_argmax = torch.max(sims, dim=1) # (M,), (M,)
        for i in row_max.argsort(descending=True):
            if row_max[i] > cfg.sim_threshold:
                assign_mat[i, row_argmax[i]] = 1
            else:
                break
    else:
        raise ValueError(f"Unknown matching method: {cfg.match_method}")
    
    return assign_mat

def prepare_objects_save_vis(objects: MapObjectList, downsample_size: float=0.025):
    objects_to_save = copy.deepcopy(objects)
            
    # Downsample the point cloud
    for i in range(len(objects_to_save)):
        objects_to_save[i]['pcd'] = objects_to_save[i]['pcd'].voxel_down_sample(downsample_size)

    # Remove unnecessary keys
    for i in range(len(objects_to_save)):
        for k in list(objects_to_save[i].keys()):
            if k not in [
                'pcd', 'bbox', 'clip_ft', 'text_ft', 'class_id', 'num_detections', 'inst_color'
            ]:
                del objects_to_save[i][k]
                
    return objects_to_save.to_serializable()
    
def process_cfg(cfg: DictConfig):
    cfg.dataset_root = Path(cfg.dataset_root)
    cfg.dataset_config = Path(cfg.dataset_config)
    
    if cfg.dataset_config.name != "multiscan.yaml":
        # For datasets whose depth and RGB have the same resolution
        # Set the desired image heights and width from the dataset config
        dataset_cfg = omegaconf.OmegaConf.load(cfg.dataset_config)
        if cfg.image_height is None:
            cfg.image_height = dataset_cfg.camera_params.image_height
        if cfg.image_width is None:
            cfg.image_width = dataset_cfg.camera_params.image_width
        print(f"Setting image height and width to {cfg.image_height} x {cfg.image_width}")
    else:
        # For dataset whose depth and RGB have different resolutions
        assert cfg.image_height is not None and cfg.image_width is not None, \
            "For multiscan dataset, image height and width must be specified"

    return cfg

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

def merge_segments_test(scene_objects, scene_name, query_feat, sim, object_pcd, object_feat, device='cuda', thresh=0.7):
    # get the indices of all objects above a threshold
    query_thresh = [i for i in range(len(sim)) if sim[i] >= thresh]

    # merge the pcds of all similar objects
    # top_pcds = [scene_objects[scene_name]['object_pcd'][idx].paint_uniform_color([1, 0, 0]) for idx in obj_obj_thresh]
    top_pcds = [scene_objects[scene_name]['object_pcd'][idx].paint_uniform_color([1, 0, 0]) for idx in query_thresh]
    # top_feats = [scene_objects[scene_name]['object_feats'][idx] for idx in obj_obj_thresh]
    top_feats = [scene_objects[scene_name]['object_feats'][idx] for idx in query_thresh]
    # append the original pcd if empty
    if len(top_pcds) == 0:
        top_pcds.append(object_pcd)
        top_feats.append(object_feat)

    # get the axis aligned bounding box for each top pcd
    top_boxes = [o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(np.asarray(pcd.points))) for pcd in top_pcds]
    # convert boxes to list of points
    top_boxes = [np.asarray(box.get_box_points()) for box in top_boxes]
    # stack 
    top_boxes = np.stack(top_boxes)
    # convert to tensor
    top_boxes = torch.from_numpy(top_boxes).to('cuda')
    # get the iou between all top boxes
    iou = compute_3d_iou_accuracte_batch(top_boxes, top_boxes)
    # set elements below the diagonal to 0
    iou = torch.triu(iou, diagonal=1)
    # return indices of ious greater than 0.5
    iou_thresh = torch.where(iou > 0.3)
    # get the unique indices
    clusters = []
    for i in range(iou_thresh[0].shape[0]):
        obj1 = iou_thresh[0][i].item()
        obj2 = iou_thresh[1][i].item()
        cluster_found = False
        for cluster in clusters:
            if obj1 in cluster:
                cluster.add(obj2)
                cluster_found = True
                break
            elif obj2 in cluster:
                cluster.add(obj1)
                cluster_found = True
                break
        if not cluster_found:
            clusters.append({obj1, obj2})

    # augment the top_pcds and top_feats using the clusters
    new_top_pcds = []
    new_top_feats = []
    for cluster in clusters:
        cluster = list(cluster)
        # merge the pcd
        merged_pcd = o3d.geometry.PointCloud()
        for idx in cluster:
            merged_pcd += top_pcds[idx]
        # merged_pcd = process_pcd(merged_pcd, cfg, run_dbscan=True)
        new_top_pcds.append(merged_pcd)
        # get the mean feature
        new_top_feats.append(torch.stack([top_feats[idx] for idx in cluster]).mean(0))
    
    # add all objects not from a cluster to the new top pcds
    for i in range(len(top_pcds)):
        if i not in [idx for cluster in clusters for idx in cluster]:
            new_top_pcds.append(top_pcds[i])
            new_top_feats.append(top_feats[i])
    
    # update the top pcds and top feats
    top_pcds = new_top_pcds
    top_feats = new_top_feats

    # recalculate the most similar object
    sim = F.cosine_similarity(query_feat, torch.stack(top_feats).to(device), dim=1)
    max_val, max_idx = sim.max(0)
    final_object_pcd = top_pcds[max_idx]
    # final_object_pcd = process_pcd(final_object_pcd, cfg, run_dbscan=True)
    final_object_feat = top_feats[max_idx]

    return final_object_pcd, final_object_feat
    
@hydra.main(version_base=None, config_path=f"{os.environ['CG_FOLDER']}/conceptgraph/configs/slam_pipeline", config_name="base")
def main(cfg : DictConfig):
    # Process the config
    cfg = process_cfg(cfg)

    # add object method to config
    object_method = cfg.object_method
    try:
        checkpoint_path = cfg.checkpoint_path
    except:
        checkpoint_path = None

    split = cfg.split

    # open the splits
    splits_path = cfg.dataset_root / 'splits' / f'nvs_sem_{split}.txt'
    # read each line and add to the splits file
    with open(splits_path) as fin:
        scenes = [line.strip() for line in fin]
    # scenes = scenes[0:2]

    # open the list of semantic classes
    semantic_fp = cfg.dataset_root / 'metadata' / 'semantic_classes.txt'
    with open(semantic_fp) as fin:
        semantic_classes = [line.strip() for line in fin]
    # open the top 100 classses
    top_fp = cfg.dataset_root / 'metadata' / 'semantic_benchmark' / 'top100.txt'
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

    # get the descriptive queries
    affordance_fp = cfg.dataset_root / 'metadata' / 'top100_affordances.txt'
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
    descriptive_fp = cfg.dataset_root / 'metadata' / 'top100_descriptions.txt'
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

    # get the index of the exclude classes in the top100
    exclude_idx = [top100.index(cls) for cls in exclude_classes]
    # remove from each list
    top100 = [top100[idx] for idx in range(len(top100)) if idx not in exclude_idx]
    affordance_list = [affordance_list[idx] for idx in range(len(affordance_list)) if idx not in exclude_idx]
    descriptive_list = [descriptive_list[idx] for idx in range(len(descriptive_list)) if idx not in exclude_idx]

    # extract the first 50 of each
    top50 = top100[0:50]
    affordance_list = affordance_list[0:50]
    descriptive_list = descriptive_list[0:50]

    # update the dictionaries
    unambiguous_affordances = {cls: affordance_queries[cls] for cls in top100}
    descriptive_queries = {cls: descriptive_queries[cls] for cls in top100}

    # define the train classes
    train_classes = top50 + affordance_list + descriptive_list
    # train_classes = train_classes[::n_core_concepts]
    print(train_classes)
    print(len(train_classes))

    # encode the train classes
    with torch.no_grad():
        caption_query = [f'an image of a {concept}' for concept in train_classes]
        caption_targets = clip_tokenizer(caption_query)
        caption_targets = caption_targets.to(device)
        core_feats = CLIP_Net.encode_text(caption_targets)

    del clip_tokenizer
    del CLIP_Net

    # augment the dataset root for saving to file
    scannet_data_root = cfg.dataset_root
    cfg['dataset_root'] = cfg['dataset_root'] / 'conceptgraph_data'

    # some hyperparameters for this algorithm
    TOP_K_SCENE = 5
    PADDING = 20

    # save path
    segment_fp = scannet_data_root / 'finetune_data' / 'preprocessed_segments' / split 
    # create this director
    os.makedirs(segment_fp, exist_ok=True)

    # get a list of the file names in the directory
    existing_files = os.listdir(segment_fp)
    existing_scenes = set([f.split('.')[0] for f in existing_files])

    # iterate through each scene and get the objects that are similar to the core classes
    for scene_name in tqdm(scenes):

        # data structure we are building
        segmented_objects = {}
        segmented_objects = {'feature': [], 'caption': [], 'scene_name': [], 'core_classes': [], 'image_path': [], 'bbox': []}


        # skip if the scene has already been processed
        if scene_name in existing_scenes:
            continue
        
        scene_objects = {}

        # dictionary to store the object features
        scene_objects = {
            'object_feats': [],
            'object_cls': [],
            'object_pcd': [],
            'captions': [],
            'image_path': [],
            'bbox': [],
            'scene_name': []
        }

        # update config
        cfg.scene_id = scene_name
        cfg.dbscan_remove_noise = False

        # Initialize the dataset
        dataset = get_scannet_dataset(
            dataconfig=cfg.dataset_config,
            start=cfg.start,
            end=cfg.end,
            stride=cfg.stride,
            basedir=scannet_data_root,
            sequence=cfg.scene_id,
            desired_height=cfg.image_height,
            desired_width=cfg.image_width,
            device="cpu",
            dtype=torch.float,
        )
        # cam_K = dataset.get_cam_K()
        
        try:
            classes, class_colors = create_or_load_colors(cfg, cfg.color_file_name)
        except:
            print(f'Missing scene {scene_name}')
            continue

        # objects = MapObjectList(device=cfg.device)
        
        if not cfg.skip_bg:
            # Handle the background detection separately 
            # Each class of them are fused into the map as a single object
            bg_objects = {
                c: None for c in BG_CLASSES
            }
        else:
            bg_objects = None

        ################## PROCESS THE RAW SEGMENTATION DATA FROM A SCENE / SCAN ####################

        for idx in trange(len(dataset)):
            # if idx>5:
            #     continue
            # get color image
            color_path = dataset.color_paths[idx]
            image_original_pil = Image.open(color_path)

            color_tensor, depth_tensor, intrinsics, *_ = dataset[idx]
            # image_rgb = cv2.cvtColor(cv2.imread(color_path), cv2.COLOR_BGR2RGB)
            # Get the RGB image
            color_np = color_tensor.cpu().numpy() # (H, W, 3)
            image_rgb = (color_np).astype(np.uint8) # (H, W, 3)
            assert image_rgb.max() > 1, "Image is not in range [0, 255]"
            
            # Get the depth image
            depth_tensor = depth_tensor[..., 0]
            depth_array = depth_tensor.cpu().numpy()

            # # display the depth and color image with matplotlib
            # from matplotlib import pyplot as plt
            # print(color_path)
            # print(dataset.poses[idx])
            # plt.figure()
            # plt.imshow(image_rgb)
            # plt.figure()
            # plt.imshow(depth_array)
            # plt.show()

            # Get the intrinsics matrix
            cam_K = intrinsics.cpu().numpy()[:3, :3]
            
            # load grounded SAM detections
            gobs = None # stands for grounded SAM observations

            color_path = Path(color_path)
            detections_path = cfg.dataset_root / cfg.scene_id / cfg.detection_folder_name / color_path.name
            detections_path = detections_path.with_suffix(".pkl.gz")
            color_path = str(color_path)
            detections_path = str(detections_path)
            
            with gzip.open(detections_path, "rb") as f:
                gobs = pickle.load(f)

            # depth_image = Image.open(depth_path)
            # depth_array = np.array(depth_image) / dataset.png_depth_scale
            # depth_tensor = torch.from_numpy(depth_array).float().to(cfg['device']).T

            # get pose, this is the untrasformed pose.
            unt_pose = dataset.poses[idx]
            unt_pose = unt_pose.cpu().numpy()
            
            # Don't apply any transformation otherwise
            adjusted_pose = unt_pose

            # print(color_path)
            # print(adjusted_pose)
            
            fg_detection_list, bg_detection_list = gobs_to_detection_list(
                cfg = cfg,
                image = image_rgb,
                depth_array = depth_array,
                cam_K = cam_K,
                idx = idx,
                gobs = gobs,
                trans_pose = adjusted_pose,
                class_names = classes,
                BG_CLASSES = BG_CLASSES,
                color_path = color_path,
            )

            # update the pcds
            for obj in fg_detection_list:
                # add object to scene pcd
                scene_objects['object_pcd'].append(np.asarray(obj['pcd'].points))
                scene_objects['object_feats'].append(obj['clip_ft'])
                scene_objects['image_path'].append(color_path)
                scene_objects['bbox'].append(obj['xyxy'])
                scene_objects['scene_name'].append(scene_name)

        # print the number of objects
        print(f'Number of objects in scene: {len(scene_objects["object_feats"])}')
        # save the pickled segments to file
        with open(segment_fp / f'{scene_name}.pkl', 'wb') as f:
            pickle.dump(scene_objects, f)
             
if __name__ == "__main__":
    main()