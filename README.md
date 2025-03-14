# QueryAdapter
Official repository for the paper [QueryAdapter: Rapid Adaptation of Vision-Language Models in Response to Natural Language Queries](https://arxiv.org/pdf/2502.18735).
This repository provides a simple example of how to run and evaluate QueryAdapter on the Scannet++ dataset.

## Dataset and Environment Set-up
1. Download [Scannet++](https://github.com/scannetpp/scannetpp)
2. Create new virtual environment following the [ConceptGraphs](https://github.com/concept-graphs/concept-graphs) installation guide
3. Some additional dependancies may need to be installed...

## Data Pre-processing
1. Download CLIP tokeniser from [here](https://github.com/openai/CLIP/blob/main/clip/bpe_simple_vocab_16e6.txt.gz) and place in src/ds/
2. Copy the files in src/concept_graphs_data/metadata/ to the metadata folder of the Scannet++ dataset
3. Navigate to concept_graphs_data where there are scripts for running ConceptGraphs and preparing data for running QueryAdapter:
```
cd src/concept_graphs_data/
```
4. Update the download paths for ConceptGraphs and Scannet++ in the files *prepare_concept_graphs_data.sh*, *generate_segment_data.sh* and *preprocess_segments.sh*
5. Run ConceptGraphs on the Scannet++ data:
```
bash prepare_conceptgraph_data.sh
```
6. Generate the segment data for running QueryAdapter:
```
bash generate_segment_data.sh
```
7. Pre-process the ConceptGraph segment data for speeding up evaluation:
```
bash preprocess_segments.sh
```
## Run QueryAdapter
1. Navigate to the training directory:
```
cd src/training/
```
2. Update the Scannet++ filepath in the file *finetune_coop.py* to match your file system
3. Run the training script to adapt the pre-trained model towards a small set of classes present in Scannet++ dataset. Model checkpoints will be saved in the *output* folder.
```
python finetune_coop.py
```

## Eval QueryAdapter
1. Navigate to the eval directory:
```
cd src/eval/
```
2. Update the dirctory of the output folder in *eval_coop_performance.sh* to match your file system
3. Run the evaluation script to produce a CSV file of the result:
```
bash eval_coop_performance.sh
```
