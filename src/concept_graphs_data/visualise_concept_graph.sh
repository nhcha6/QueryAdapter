
export CG_FOLDER=/home/nicolas/Documents/ConceptGraphClustering/concept-graphs
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/nicolas/miniforge3/lib/

# cd ${CG_FOLDER}/conceptgraph

SCENE_NAME=95d525fbfd

python visualize_cfslam_results.py --result_path /home/nicolas/hpc-home/Datasets/scannetpp/data_download/complete_dataset/conceptgraph_data/${SCENE_NAME}/pcd_saves/full_pcd_none_overlap_maskconf0.95_simsum1.2_dbscan.1_merge20_masksub_post.pkl.gz