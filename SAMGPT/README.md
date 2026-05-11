# SAMGPT
We provide the code (in pytorch) and datasets for our paper [**"SAMGPT: Text-free Graph Foundation Model for Multi-domain Pre-training and Cross-domain Adaptation"**](https://arxiv.org/abs/2502.05424), 
which is accepted by WWW2025.

## Description

The repository is organised as follows:

- **data/**: contains data we use.
- **cache/**: stores intermediate results during pre-training.
- **checkpoints/**: contains checkpoints of the models after pre-training.
- **result/**: stores results when downstream task is done.
- **src/**: implements pre-training and downstream task.

## Package Dependencies

- python 3.8.16
- pytorch 1.10.1
- cuda 11.3
- pyG 2.1.0

## Running experiments
### Data Preparation
To download the datasets and generate fewshot data, run `python ./src/utils/generate_fewshot.py`

### Node Classification and Graph Classification
To train and evaluate SAMGPT on various target domains, you may need to change the corresponding parameters in *execute.py*.

Pre-training and prompt tuning:
```bash
python ./src/execute.py 
  --dataset {Name of the downstream dataset, default: Cora} 
  --pretrain_datasets {List of pretrain datasets, default: ['Citeseer', 'Pubmed', 'Photo', 'Computers', 'FacebookPagePage', 'LastFMAsia']} 
  --downstream_task {Type of downstream task, options: node or graph, default: node}
  --gpu {GPU device number, default: 0} 
  --shot_num {Number of examples for few-shot learning, default: 1} 
  --pretrain_method {Pretraining method, options: GRAPHCL, LP, or splitLP, default: GRAPHCL}  
```


