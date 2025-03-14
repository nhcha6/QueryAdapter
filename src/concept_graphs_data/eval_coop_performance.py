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
from clip_coop import CoOpCLIP

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

    # get the number of core concepts
    n_core_concepts = int(cfg.n_core_concepts)
    use_affordances = bool(cfg.use_affordances)

    # open the splits
    splits_path = cfg.dataset_root / 'splits' / 'nvs_sem_val.txt'
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
    # print(train_classes)

    # get the index of the exclude classes in the top100
    exclude_idx = [top100.index(cls) for cls in exclude_classes]
    # remove from each list
    top100 = [top100[idx] for idx in range(len(top100)) if idx not in exclude_idx]
    affordance_list = [affordance_list[idx] for idx in range(len(affordance_list)) if idx not in exclude_idx]
    descriptive_list = [descriptive_list[idx] for idx in range(len(descriptive_list)) if idx not in exclude_idx]

    # update the dictionaries
    unambiguous_affordances = {cls: affordance_queries[cls] for cls in top100}
    descriptive_queries = {cls: descriptive_queries[cls] for cls in top100}

    # extract the first 50 of each
    top50 = top100[0:50]
    affordance_list = affordance_list[0:50]
    descriptive_list = descriptive_list[0:50]

    # define the train classes
    all_classes = top50 + affordance_list + descriptive_list
    # train_classes = top50[1::n_core_concepts] + affordance_list[0::n_core_concepts] + descriptive_list[1::n_core_concepts]  
    # train_classes = train_classes[0::n_core_concepts]
    # train_classes = top50[n_core_concepts::10]

    if use_affordances:
        # train_classes = affordance_list[n_core_concepts::10]
        train_classes = top100_affordances[0:48][n_core_concepts::8] #+ top100_affordances[0::n_core_concepts] + top100_descriptions[1::n_core_concepts]
    else:
        # train_classes = top50[n_core_concepts::10]
        train_classes = top100[0:48][n_core_concepts::8] #+ top100_affordances[0::n_core_concepts] + top100_descriptions[1::n_core_concepts]

    print(train_classes)
    print(len(train_classes))

    # augment the dataset root for saving to file
    scannet_data_root = cfg.dataset_root
    cfg['dataset_root'] = cfg['dataset_root'] / 'conceptgraph_data'

    # declare the segments data
    segments_fp = scannet_data_root / 'finetune_data' / 'preprocessed_segments' / 'val'

    adapter_cfg = mch({
        'classnames': all_classes,
        'embedding_dim': 1024,
        'logit_scale': np.log(50),
    })
    # attempt to load the checkpoint
    if checkpoint_path is not None:
        adapter = CoOpCLIP(adapter_cfg, CLIP_Net, device, torch.float32, clip_tokenizer)
        # adapter = CustomCLIP(adapter_cfg, CLIP_Net, device)
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        # print(checkpoint)
        # print(adapter.adapter.ctx)
        adapter.adapter.ctx = checkpoint
        
        # get the text features
        coop_text_feats = adapter.return_clip_feats()
        print(coop_text_feats.shape)
        print('Adapter loaded')
    else:
        adapter = None

    incorrect_class = {'top100': 0, 'affordances': 0, 'descriptive':0, 'seen':0, 'unseen': 0}
    correct_class = {'top100': 0, 'affordances': 0, 'descriptive':0, 'seen':0, 'unseen': 0}
    for scene_name in tqdm(scenes):
        # # not currently assessed by Concepptgraphs
        # if scene_name in ['3864514494', '5942004064', '578511c8a9']:
        #     continue

        # make sure the scene has been processed by conceptgraphs
        try:
            # open the pcd file generate using conceptgraph
            pcd_path = f'{scannet_data_root}/conceptgraph_data/{scene_name}/pcd_saves/full_pcd_none_overlap_maskconf0.95_simsum1.2_dbscan.1_merge20_masksub_post.pkl.gz'
            # get the objects
            fg_objects, bg_objects, class_colors = load_result(pcd_path)
            # print(f'Loaded conceptgraph with {len(fg_objects)} objects')
        except Exception as e:
            # print(e)
            continue
        
        scene_objects = {}
        scene_gt_queries = {}

        # dictionary to store the object features
        scene_objects[scene_name] = {
            'object_feats': [],
            'object_cls': [],
            'object_pcd': []
        }
        # dictionary to store the gt queries
        scene_gt_queries[scene_name] = {}
        gt_boxes, gt_labels, gt_pcd, pcd_disp, gt_annos = get_gt_boxes(scannet_data_root, scene_name, semantic_classes)
        gt_labels = [label for label in gt_labels if label in top50]
        scene_gt_queries[scene_name]['gt_labels'] = gt_labels
        scene_gt_queries[scene_name]['gt_boxes'] = gt_boxes
        scene_gt_queries[scene_name]['gt_annos'] = gt_annos
        scene_gt_queries[scene_name]['gt_pcd'] = gt_pcd
        scene_gt_queries[scene_name]['pcd_disp'] = pcd_disp
        
        # eval using segments
        if object_method == 'segments':
            # open the segments file
            scene_fp = segments_fp / f'{scene_name}.pkl'
            with open(scene_fp, 'rb') as fin:
                segments = pickle.load(fin)

            # update the pcds
            for idx in range(len(segments['object_feats'])):
                pcd_points = segments['object_pcd'][idx]
                feat = segments['object_feats'][idx]
                scene_objects[scene_name]['object_pcd'].append(pcd_points)
                scene_objects[scene_name]['object_feats'].append(feat)

                # # visualize the object pcd
                # obj_pcd_disp = obj['pcd']
                # # set color to red
                # obj_pcd_disp.paint_uniform_color([1, 0, 0])
                # o3d.visualization.draw_geometries([obj_pcd_disp, pcd_disp])
        
        # instead use conceptgraphs
        elif object_method == 'conceptgraphs':
            for idx, instance in enumerate(fg_objects):
                scene_objects[scene_name]['object_pcd'].append(np.asarray(instance['pcd'].points))
                scene_objects[scene_name]['object_feats'].append(instance['clip_ft'])

        # run the eval
        for scene_name in scene_gt_queries.keys():
            # get all the objects in this scene
            object_feats = torch.stack(scene_objects[scene_name]['object_feats'])
            # object_cls = scene_objects[scene_name]['object_cls']
            # all_mapped_classes = [item for sublist in object_cls for item in sublist]

            # get the gt info
            gt_disp = scene_gt_queries[scene_name]['pcd_disp']
            # get the gt info
            gt_pcd = scene_gt_queries[scene_name]['gt_pcd']
            pcd_disp = scene_gt_queries[scene_name]['pcd_disp']
            gt_annos = scene_gt_queries[scene_name]['gt_annos']
            
            for query_class in set(scene_gt_queries[scene_name]['gt_labels']):
                # query features
                query_features = []
                gt_queries = []
                
                # generate feature
                text_caption = [f'an image of a {query_class}']
                with torch.no_grad():
                    caption_targets = clip_tokenizer(text_caption)
                    caption_targets = caption_targets.to(device)
                    caption_feat = CLIP_Net.encode_text(caption_targets)
                    query_features.append(caption_feat)
                    gt_queries.append(query_class)

                # if this object has an affordance, process it
                if query_class in unambiguous_affordances.keys():
                    aff_feat = affordance_features[query_class]
                    query_features.append(aff_feat)
                    gt_queries.append(unambiguous_affordances[query_class])

                # if this object has a descriptive query, process it
                if query_class in desciptive_features.keys():
                    desc_feat = desciptive_features[query_class]
                    query_features.append(desc_feat)
                    gt_queries.append(descriptive_queries[query_class])
                
                for i in range(len(query_features)):
                    query_feat = query_features[i]
                    gt_query = gt_queries[i]

                    # skip all but the gt_Query
                    if gt_query not in train_classes:
                        continue

                    # get similarity between the query and the object features
                    object_feats = object_feats.to(device)
                    query_feat = query_feat.to(device)
                    # ensure it sums to 1
                    object_feats = object_feats / object_feats.norm(dim=-1, keepdim=True)
                    query_feat = query_feat / query_feat.norm(dim=-1, keepdim=True)
                    
                    # get the logits
                    if adapter is not None:
                        # get the index in train classes
                        idx = all_classes.index(gt_query)
                        # get the correspoding coop text feature
                        coop_text_feat = coop_text_feats[idx]
                        sim = F.cosine_similarity(coop_text_feat, object_feats, dim=1)
                    else:
                        # get the cosine similarity
                        sim = F.cosine_similarity(query_feat, object_feats, dim=1)

                    # get the maximum value
                    max_val, max_idx = sim.max(0)
                    # find the gt class of the matched object
                    object_pcd = scene_objects[scene_name]['object_pcd'][max_idx]
                    object_feat = scene_objects[scene_name]['object_feats'][max_idx].to(device)

                    # # get the sim between the object feat and all other objects
                    # obj_obj_sim = F.cosine_similarity(object_feat.unsqueeze(0), object_feats, dim=1)

                    # # get indices of all objects that have similar features to the best match (as a second check, we can assess spatial similarity as well)
                    # obj_obj_thresh = [i for i in range(len(obj_obj_sim)) if obj_obj_sim[i] >= 0.8]
                
                    # method for merging objects that are similar to the query and returning the best match.
                    # final_object_pcd, final_object_feat = merge_segments_test(scene_objects, scene_name, query_feat, sim, object_pcd, object_feat, device='cuda', thresh=0.7)

                    # # merge the pcd
                    # merged_pcd = o3d.geometry.PointCloud()
                    # for pcd in top_pcds:
                    #     merged_pcd += pcd
                    # merged_pcd = process_pcd(merged_pcd, cfg, run_dbscan=True)
                    # object_pcd = np.asarray(final_object_pcd.points)

                    # match the pcd_obj to the gt_pcd
                    pred_gt_associations = compute_pred_gt_associations(torch.tensor(object_pcd).unsqueeze(0).cuda().contiguous().float(), torch.tensor(gt_pcd).unsqueeze(0).cuda().contiguous().float()) 
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
                        gt_labels = [gt_class]
                    except:
                        gt_class = []
                
                    # # get the ratio of the max value to the total
                    # ratios =[val/pred_gt_associations[0].shape[0] for val in counts]
                    # # get indices where ratio is above a threshold
                    # thresh = [i for i in range(len(ratios)) if ratios[i] > 0.5]
                    # # get the labels of the above threshold
                    # gt_labels = [unique[i] for i in thresh if unique[i] in target_classes]

                    # add the affordances to the gt_labels
                    for gt_label in gt_labels:
                        if gt_label in unambiguous_affordances.keys():
                            gt_labels.append(unambiguous_affordances[gt_label])
                    
                        # add the descriptive queries to the gt_labels
                        if gt_label in descriptive_queries.keys():
                            gt_labels.append(descriptive_queries[gt_label])

                    # update the results for top50, affordance, long-tail
                    if gt_query in train_classes:
                        if gt_query in gt_labels:
                            if gt_query in top100:
                                correct_class['top100'] += 1
                            elif gt_query in unambiguous_affordances.values():
                                correct_class['affordances'] += 1
                            elif gt_query in descriptive_queries.values():
                                correct_class['descriptive'] += 1
                            # else:
                            #     correct_class['long-tail'] += 1
                        else:
                            if gt_query in top100:
                                incorrect_class['top100'] += 1
                            elif gt_query in unambiguous_affordances.values():
                                incorrect_class['affordances'] += 1
                            elif gt_query in descriptive_queries.values():
                                incorrect_class['descriptive'] += 1
                            # else:
                            #     incorrect_class['long-tail'] += 1
                    
                    # pdate the results for seen and unseen classes
                    if gt_query in gt_labels:
                        if gt_query in train_classes:
                            correct_class['seen'] += 1
                        else:
                            correct_class['unseen'] += 1
                    else:
                        if gt_query in train_classes:
                            incorrect_class['seen'] += 1
                        else:
                            incorrect_class['unseen'] += 1

                    # print(f'Query: {gt_query}')
                    # print(f'Pred: {gt_labels}')

                    # display the top 10 pcds
                    # o3d.visualization.draw_geometries([final_object_pcd, gt_disp])

                    # # display the selected object pcd red on the gt pcd
                    # obj_pcd = scene_objects[scene_name]['object_pcd'][max_idx]
                    # obj_pcd.paint_uniform_color([1, 0, 0])
                    # o3d.visualization.draw_geometries([obj_pcd, gt_disp])
        
        try:
            # # print current results
            # # print('top50 queries: ', correct_class['top50'] + incorrect_class['top50'])
            # # print('top50 recall: ', correct_class['top50'] / (correct_class['top50'] + incorrect_class['top50']))
            # print('top100 queries: ', correct_class['top100'] + incorrect_class['top100'])
            # print('top100 recall: ', correct_class['top100'] / (correct_class['top100'] + incorrect_class['top100']))
            # # print('long-tail queries: ', correct_class['long-tail'] + incorrect_class['long-tail'])
            # # print('long-tail recall: ', correct_class['long-tail'] / (correct_class['long-tail'] + incorrect_class['long-tail']))
            # print('affordances queries: ', correct_class['affordances'] + incorrect_class['affordances'])
            # print('affordances recall: ', correct_class['affordances'] / (correct_class['affordances'] + incorrect_class['affordances']))
            # print('descriptive queries: ', correct_class['descriptive'] + incorrect_class['descriptive'])
            # print('descriptive recall: ', correct_class['descriptive'] / (correct_class['descriptive'] + incorrect_class['descriptive']))
            # print('\n')
            # print('seen queries: ', correct_class['seen'] + incorrect_class['seen'])
            # print('seen recall: ', correct_class['seen'] / (correct_class['seen'] + incorrect_class['seen']))
            # print('unseen queries: ', correct_class['unseen'] + incorrect_class['unseen'])
            # print('unseen recall: ', correct_class['unseen'] / (correct_class['unseen'] + incorrect_class['unseen']))
            # # print('total class queries: ', correct_class['top50'] + incorrect_class['top50'] + correct_class['long-tail'] + incorrect_class['long-tail'])
            # # print('total class recall: ', (correct_class['top50'] + correct_class['long-tail']) / (correct_class['top50'] + incorrect_class['top50'] + correct_class['long-tail'] + incorrect_class['long-tail']))
            
            # create the eval file
            eval_file = cfg.save_name
            # write the results to csv
            with open(eval_file, 'w') as file:
                # file.write('top100 queries, top100 recall, affordances queries, affordances recall, descriptive queries, descriptive recall, seen queries, seen recall, unseen queries, unseen recall\n')
                # file.write(f'{correct_class["top100"] + incorrect_class["top100"]}, {correct_class["top100"] / (correct_class["top100"] + incorrect_class["top100"])}, {correct_class["affordances"] + incorrect_class["affordances"]}, {correct_class["affordances"] / (correct_class["affordances"] + incorrect_class["affordances"])}, {correct_class["descriptive"] + incorrect_class["descriptive"]}, {correct_class["descriptive"] / (correct_class["descriptive"] + incorrect_class["descriptive"])}, {correct_class["seen"] + incorrect_class["seen"]}, {correct_class["seen"] / (correct_class["seen"] + incorrect_class["seen"])}, {correct_class["unseen"] + incorrect_class["unseen"]}, {correct_class["unseen"] / (correct_class["unseen"] + incorrect_class["unseen"])}\n')
                # file.write(f'top100_queries, {correct_class["top100"] + incorrect_class["top100"]}\n')
                # file.write(f'top100_recall, {correct_class["top100"] / (correct_class["top100"] + incorrect_class["top100"])}\n')
                # file.write(f'affordances_queries, {correct_class["affordances"] + incorrect_class["affordances"]}\n')
                # file.write(f'affordances_recall, {correct_class["affordances"] / (correct_class["affordances"] + incorrect_class["affordances"])}\n')
                # file.write(f'descriptive_queries, {correct_class["descriptive"] + incorrect_class["descriptive"]}\n')
                # file.write(f'descriptive_recall, {correct_class["descriptive"] / (correct_class["descriptive"] + incorrect_class["descriptive"])}\n')
                file.write(f'seen_queries, {correct_class["seen"] + incorrect_class["seen"]}\n')
                file.write(f'seen_recall, {correct_class["seen"] / (correct_class["seen"] + incorrect_class["seen"])}\n')
                # file.write(f'unseen_queries, {correct_class["unseen"] + incorrect_class["unseen"]}\n')
                # file.write(f'unseen_recall, {correct_class["unseen"] / (correct_class["unseen"] + incorrect_class["unseen"])}\n')

        except:
            pass

if __name__ == "__main__":
    main()