## Understanding the Role of Structure in Legal Contract Clause Classification

This repository contains all the code used for this thesis.

## Getting Started

To get a local copy up and running follow these simple example steps.

### Installation
1. Clone the repo
   ```sh
   git clone https://github.com/elizabethzhong/thesis
   ```
2. Install pip3 packages
   ```sh
   pip3 install -r requirements.txt 
   ```
## Usage
This section details the minimum steps required to generate the knowledge graph, embeddings and finally train and evaluate a classifier for important labels in the [CUAD dataset](https://arxiv.org/abs/2103.06268). 

1. Generate knowledge graph from dataset.
```sh
python3 run.py
```
2. Run dgi on knowledge graph to generate embeddings.
Requires .gpickle file from previous step
```sh
python3 -m dgi.trainer
```
3. Run classifier.
Requires .gpickle and .pkl file from previous steps
```sh
python3 trainerClassification.py
```
Note: Each programs requires file generated from the previous file. The file names in the code are set to work without any modifications however if the paths for exported files are modified, it must be modified in the code for subsequent steps. 