# QueryAdapter

## Dataset and Environment Set-up
1. Download Scannet++
2. Create new virtual environment and follow ConceptGraphs installation guide

## Data Pre-processing
1. Run scripts.....
2. Download CLIP tokeniser from [here](https://github.com/openai/CLIP/blob/main/clip/bpe_simple_vocab_16e6.txt.gz) and place in src/ds/
3. Copy the files in src/concept_graphs_data/metadata/ to the metadata folder of the Scannet++ dataset

## Run QueryAdapter
1. Navigate to the training directory:
```
cd src/training/
```
2. Update the Scannet++ filepath in the file finetune_coop.py to match your file system
3. Run the script:
```
python finetune_coop.py
```

## Eval QueryAdapter
1. Navigate to the eval directory:
```
cd src/eval/
```
2. Update the dirctory of the output folder in eval_coop_performance.sh to match your file system
3. Run the script:
```
bash eval_coop_performance.sh
```
