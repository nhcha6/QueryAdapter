export GSA_PATH=/home/nicolas/Documents/ConceptGraphClustering/Grounded-Segment-Anything
export SCANNET_ROOT=/home/nicolas/hpc-home/Datasets/scannetpp/data_download/complete_dataset/
export CG_FOLDER=/home/nicolas/Documents/ConceptGraphClustering/concept-graphs
export SCANNET_CONFIG_PATH=scannetpp_config.yaml
export THRESHOLD=1.2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/nicolas/miniforge3/lib/

# cd ${CG_FOLDER}/conceptgraph

results_path='../../results_adapters/10_Adapter_Affk_Top50/output/'
method='segments'
model='epoch200'
use_affordances=True

# list all folders in the current directory
for folder in $results_path*; do
    echo $folder

    # folder='../../ckpt/17_Segments_Data_Size/segments_medium_0.0005_512_4'

    # split folder name at _ and get the last element (should always be the number of core classes to assess)
    arrIN=(${folder//_/ })
    core_num=${arrIN[-1]}
    use_affordances=${arrIN[-2]}

    # python eval_segment_performance.py \
    #     dataset_root=$SCANNET_ROOT \
    #     dataset_config=$SCANNET_CONFIG_PATH \
    #     stride=5 \
    #     scene_id=$SCENE_NAME \
    #     spatial_sim_type=overlap \
    #     mask_conf_threshold=0.95 \
    #     match_method=sim_sum \
    #     sim_threshold=${THRESHOLD} \
    #     dbscan_eps=0.1 \
    #     gsa_variant=none \
    #     class_agnostic=True \
    #     skip_bg=True \
    #     max_bbox_area_ratio=0.5 \
    #     save_suffix=overlap_maskconf0.95_simsum${THRESHOLD}_dbscan.1_merge20_masksub_test \
    #     merge_interval=20 \
    #     merge_visual_sim_thresh=0.8 \
    #     merge_text_sim_thresh=0.8 \
    #     +object_method=${method} \
    #     +n_core_concepts=$core_num \
    #     +save_name=${folder}/eval_${method}_pretrained.csv \
        # +use_affordances=${use_affordances} \
    #     # +checkpoint_path=${train_path}/${folder}/best.pth

    python eval_segment_performance.py \
        dataset_root=$SCANNET_ROOT \
        dataset_config=$SCANNET_CONFIG_PATH \
        stride=5 \
        scene_id=0a7cc12c0e \
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
        merge_text_sim_thresh=0.8 \
        +object_method=${method} \
        +n_core_concepts=${core_num} \
        +save_name=${folder}/eval_${method}_${model}.csv \
        +use_affordances=${use_affordances} \
        +checkpoint_path=${folder}/${model}.pth

    done

python average_results.py --folder ${results_path} --method ${method} --model ${model}