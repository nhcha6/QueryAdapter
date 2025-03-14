#!/bin/bash

# !! Update for your download paths of CG and Scannet
export GSA_PATH=/home/nicolas/Documents/ConceptGraphClustering/Grounded-Segment-Anything
export SCANNET_ROOT=/home/nicolas/hpc-home/Datasets/scannetpp/data_download/complete_dataset/
export CG_FOLDER=/home/nicolas/Documents/ConceptGraphClustering/concept-graphs
export SCANNET_CONFIG_PATH=scannetpp_config.yaml

# cd ${CG_FOLDER}/conceptgraph

SCENE_NAME=0a7cc12c0e

# The CoceptGraphs (without open-vocab detector)
python generate_gsa_results.py \
        --dataset_root $SCANNET_ROOT \
        --dataset_config $SCANNET_CONFIG_PATH \
        --scene_id $SCENE_NAME \
        --class_set none \
        --stride 5

# # The CoceptGraphs (without open-vocab detector)
# python generate_gsa_results.py \
#     --dataset_root $SCANNET_ROOT \
#     --dataset_config $SCANNET_CONFIG_PATH \
#     --scene_id $SCENE_NAME \
#     --class_set none \
#     --stride 5

# # The ConceptGraphs-Detect 
# CLASS_SET=ram
# python scripts/generate_gsa_results.py \
#     --dataset_root $REPLICA_ROOT \
#     --dataset_config $REPLICA_CONFIG_PATH \
#     --scene_id $SCENE_NAME \
#     --class_set $CLASS_SET \
#     --box_threshold 0.2 \
#     --text_threshold 0.2 \
#     --stride 5 \
#     --add_bg_classes \
#     --accumu_classes \
#     --exp_suffix withbg_allclasses