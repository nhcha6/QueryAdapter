export GSA_PATH=/home/nicolas/Documents/ConceptGraphClustering/Grounded-Segment-Anything
export SCANNET_ROOT=/home/nicolas/hpc-home/Datasets/scannetpp/data_download/complete_dataset/
export CG_FOLDER=/home/nicolas/Documents/ConceptGraphClustering/concept-graphs
export SCANNET_CONFIG_PATH=scannetpp_config.yaml
export THRESHOLD=1.2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/nicolas/miniforge3/lib/

# cd ${CG_FOLDER}/conceptgraph

SCENE_NAME=0a7cc12c0e

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
    save_suffix=overlap_maskconf0.95_simsum${THRESHOLD}_dbscan.1_merge20_masksub_test \
    merge_interval=20 \
    merge_visual_sim_thresh=0.8 \
    merge_text_sim_thresh=0.8