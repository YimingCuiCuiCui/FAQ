# FAQ: Feature Aggregated Queries for Transformer-based Video Object Detectors


This repository is an official implementation of the paper [Feature Aggregated Queries for Transformer-based Video Object Detectors].


## Installation

The codebase is built on top of [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR).

### Requirements

* Linux, CUDA>=9.2, GCC>=5.4
  
* Python>=3.7

    We recommend you to use Anaconda to create a conda environment:
    ```bash
    conda create -n FAQ python=3.7 pip
    ```
    Then, activate the environment:
    ```bash
    conda activate FAQ
    ```
  
* PyTorch>=1.5.1, torchvision>=0.6.1 (following instructions [here](https://pytorch.org/)

    For example, if your CUDA version is 9.2, you could install pytorch and torchvision as following:
    ```bash
    conda install pytorch=1.5.1 torchvision=0.6.1 cudatoolkit=9.2 -c pytorch
    ```
  
* Other requirements
    ```bash
    pip install -r requirements.txt
    ```

* Build MultiScaleDeformableAttention
    ```bash
    cd ./models/ops
    sh ./make.sh
    ```

## Usage

### Dataset preparation

1. Please download ILSVRC2015 DET and ILSVRC2015 VID dataset from [here](https://image-net.org/challenges/LSVRC/2015/2015-downloads). Then we covert jsons of two datasets by using the [code](https://github.com/open-mmlab/mmtracking/blob/master/tools/convert_datasets/ilsvrc/). The joint [json](https://drive.google.com/drive/folders/1cCXY41IFsLT-P06xlPAGptG7sc-zmGKF?usp=sharing)  of two datasets is provided. The  After that, we recommend to symlink the path to the datasets to datasets/. And the path structure should be as follows:

```
code_root/
└── data/
    └── vid/
        ├── Data
            ├── VID/
            └── DET/
        └── annotations/
        	  ├── imagenet_vid_train.json
            ├── imagenet_vid_train_joint_30.json
        	  └── imagenet_vid_val.json

```

### Training
We use ResNet50 and ResNet101 as the network backbone. We train our FAQ with ResNet50 as backbone as following:

#### Training on single node
1. Train SingleBaseline. You can download COCO pretrained weights from [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR). 
   
```bash 
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh $1 r50 $2 configs/r50_train_single.sh
```  
1. Train FAD. Using the model weights of SingleBaseline as the resume model.

```bash 
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh $1 r50 $2 configs/r50_train_multi.sh
``` 


#### Training on slurm cluster
If you are using slurm cluster, you can simply run the following command to train on 1 node with 8 GPUs:
```bash
GPUS_PER_NODE=8 ./tools/run_dist_slurm.sh <partition> r50 8 configs/r50_train_multi.sh
```

### Evaluation
You can get the config file and pretrained model of FAQ (the link is in "Main Results" session), then put the pretrained_model into correponding folder.
```
code_root/
└── exps/
    └── our_models/
        ├── COCO_pretrained_model
        ├── exps_single
        └── exps_multi
```
And then run following command to evaluate it on ImageNET VID validation set:
```bash 
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh $1 eval_r50 $2 configs/r50_eval_multi.sh
```



## Citing FAQ
If you find FAQ useful in your research, please consider citing:
```bibtex
@misc{cui2023faq,
      title={FAQ: Feature Aggregated Queries for Transformer-based Video Object Detectors}, 
      author={Yiming Cui and Linjie Yang},
      year={2023},
      eprint={2303.08319},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
