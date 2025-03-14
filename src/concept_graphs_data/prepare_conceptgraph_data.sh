#!/bin/bash

# UPDATE YOUR DIRECTORIES FOR CONCEPTGRAPH AND SCANNET HERE!
export GSA_PATH=/home/nicolas/Documents/ConceptGraphClustering/Grounded-Segment-Anything
export SCANNET_ROOT=/home/nicolas/hpc-home/Datasets/scannetpp/data_download/complete_dataset/
export CG_FOLDER=/home/nicolas/Documents/ConceptGraphClustering/concept-graphs
export SCANNET_CONFIG_PATH=scannetpp_config.yaml
export THRESHOLD=1.2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/nicolas/miniforge3/lib/

# cd ${CG_FOLDER}/conceptgraph

# all scenes
search_dir=${SCANNET_ROOT}data
for file in "$search_dir"/*
do
    SCENE_NAME=$(basename ${file})
# # manually defined scenes
# for SCENE_NAME in 3864514494 5942004064 578511c8a9
# do
    echo $SCENE_NAME
    
    # The ConceptGraphs (without open-vocab detector)
    python generate_gsa_results.py \
            --dataset_root $SCANNET_ROOT \
            --dataset_config $SCANNET_CONFIG_PATH \
            --scene_id $SCENE_NAME \
            --class_set none \
            --stride 5
    
    python cfslam_pipeline_batch.py \
        dataset_root=$SCANNET_ROOT \
        dataset_config=$SCANNET_CONFIG_PATH \
        stride=5 \
        scene_id=$SCENE_NAME \
        spatial_sim_type=overlap \
        mask_conf_threshold=0.95 \
        match_method=sim_sum \
        sim_threshold=${THRESHOLD} \
        dbscan_eps=0.1 \
        gsa_variant=none \
        class_agnostic=True \
        skip_bg=True \
        max_bbox_area_ratio=0.5 \
        save_suffix=overlap_maskconf0.95_simsum${THRESHOLD}_dbscan.1_merge20_masksub \
        merge_interval=20 \
        merge_visual_sim_thresh=0.8 \
        merge_text_sim_thresh=0.8
        
done

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