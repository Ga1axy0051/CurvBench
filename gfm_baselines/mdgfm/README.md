# MDGFM
<h1 align="center"> Multi-Domain Graph Foundation Models: Robust Knowledge Transfer via Topology Alignment </a></h2>



## Setup Environment

- python 3.9.20
- pytorch 1.10.1+cu113
- torch_cluster 1.6.0
- torch_geometric 2.1.0
- torch_scatter 2.0.9
- torch_sparse 0.6.13
- torch_spline_conv 1.2.1
- dgl 0.9.1
- cuda 11.3
- pyG 2.1.0
  
## Running experiments

'preprompt.py' is the code for the pre training phase, 'downprompt.py' is downstream code and 'MDGFM.py' is the entire code of our model. 

### For different datasets, please run the following code：
> one-shot
```
python runexp.py --dataset **** --drop_percent 0.5 --lr ** --downstreamlr ** --epochs ** --shot_num 1
```
> few-shot
```
python runexp.py --dataset **** --drop_percent 0.5 --lr ** --downstreamlr ** --epochs ** --shot_num 5
```

### For example:
> Cora:
```
python runexp.py --dataset Cora --drop_percent 0.5 --lr 0.0075 --downstreamlr 0.001 --epochs 60 --shot_num 1
```