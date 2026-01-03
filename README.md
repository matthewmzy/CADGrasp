# CADGraspNet 2.0
Official code for "**CADGrasp: Learning Contact and Collision Aware General Dexterous Grasping in Cluttered Scenes **" *(NeurIPS 2025)*

[Project Page](https://https://cadgrasp.github.io/) | [Paper](https://arxiv.org/pdf/2410.23004)

![image](./figure/teaser.png)

## Environment

- Ubuntu 22.04
- CUDA 12.1


```bash

conda create -n CADGrasp python=3.8
conda activate CADGrasp

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch3d/linux-64/pytorch3d-0.7.5-py38_cu117_pyt201.tar.bz2
conda install -y --use-local ./pytorch3d-0.7.5-py38_cu117_pyt201.tar.bz2

git clone git@github.com:wrc042/TorchSDF.git
(cd TorchSDF; pip install -e .)

git clone git@github.com:mzhmxzh/torchprimitivesdf.git
(cd torchprimitivesdf; pip install -e .)

# Download IsaacGym4 from https://developer.nvidia.com/isaac-gym
(cd isaacgym/python; pip install -e .)

pip install plotly

pip install transforms3d

pip install open3d==0.17.0

pip install urdf_parser_py

pip install tensorboard

pip install coacd

pip install rich

pip install ikpy

pip install einops

git clone git@github.com:huggingface/diffusers.git
(cd diffusers; pip install -e ".[torch]")

pip install graspnetAPI

pip install wandb

# wandb login
# enter the API key when prompted
# you can also use WANDB_MODE=offline in training if you don't need logging

git clone https://github.com/NVIDIA/MinkowskiEngine.git
sudo apt install libopenblas-dev
export CUDA_HOME=/usr/local/cuda-11.7
(cd MinkowskiEngine; python setup.py install --blas=openblas)

git clone https://github.com/nkolot/nflows.git
pip install -e nflows/

pip install numpy==1.23.0 # You can ignore the version conflict between graspnetAPI and numpy
# for gripper experiments, if the ap result is significantly low, there might be a bug in graspnetapi's np.matmul. please update numpy to 1.24.1 and replace the np.float to float whenever there is AttributeError: module 'numpy' has no attribute 'float' and all np.int to int whenever there is AttributeError: module 'numpy' has no attribute 'int'. Only two files need to be modified
# this might happen in some cpu

pip install PyOpenGL
pip install glfw
pip install pyglm

pip install healpy
pip install rtree
```

## Data

Download data from https://huggingface.co/datasets/lhrlhr/DexGraspNet2.0, then unzip them and put them in the data directory.

Users from Chinese mainland can download using mirrors like https://hf-mirror.com/

The data architecture should be:

```
data/
    meshdata/
    acronym_test_scenes/
    scenes/
    dex_graspness_new/ (you can also generate using src/preprocess/dex_graspness.py)
    dex_grasps_new/
    gripper_graspness/
    gripper_grasps/
    meshdata/
    models/ (link to meshdata)
```

## Checkpoints

Download the checkpoints in the dataset link.

## Preprocessing


```bash
# Gripper (you can download gripper_grasps and gripper_graspness instead)
python src/preprocess/extract_gripper_grasp.py --start 0 --end 100 # require graspnet data
python src/preprocess/refine_dataset.py
python src/preprocess/gripper_graspness.py --start 0 --end 100
# Dexterous hand: compute graspness (you can download dex_graspness_new and dex_grasps_new instead)
python src/preprocess/dex_graspness.py --start 0 --end 100
python src/preprocess/dex_graspness.py --start 1000 --end 8500 # split this if you have multiple GPUs
```

```bash
# compute edges for evaluation
python src/preprocess/compute_edges.py --dataset graspnet --start 100 --end 190
python src/preprocess/compute_edges.py --dataset graspnet --start 200 --end 380
python src/preprocess/compute_edges.py --dataset graspnet --start 9000 --end 9900
python src/preprocess/compute_edges.py --dataset acronym 
```

```bash
# collect network input for evaluation 
python src/preprocess/compute_network_input_all.py --dataset graspnet --scene_id_start 100 --scene_id_end 190
python src/preprocess/compute_network_input_all.py --dataset graspnet --scene_id_start 200 --scene_id_end 380
python src/preprocess/compute_network_input_all.py --dataset graspnet --scene_id_start 9000 --scene_id_end 9900
python src/preprocess/compute_network_input_all.py --dataset acronym 
```

```bash
bash IBSProcessing/scripts/evaluate_grasps.py
bash IBSProcessing/scripts/batch_cal_ibs.py
bash IBSProcessing/scripts/annotate_ibs_for_view.py
```

## Training

```bash
cd LASDiffusion
bash train.sh
```

## Evaluation

```bash
python src/eval/predict_dexterous_all_cates.py --ckpt experiments/cad/ckpt/ckpt_50000.pth 
```

```bash
# evaluate dexterous grasping poses in IsaacGym
python src/eval/evaluate_dexterous_all_cates.py # fill the ckpt path in ckpt_path_list in evaluate_dexterous_all.py. It is quicker to evaluate multiple checkpoints together
```

```bash
# print the dexterous grasping's simulation result
python src/eval/print_dexterous_result.py --ckpt experiments/cad/ckpt/ckpt_50000.pth
```


## Citation

```
@inproceedings{zhang2024CADGraspnet,
  title={CADGrasp: Learning Contact and Collision Aware General Dexterous Grasping in Cluttered Scenes},
  author={Zhang, Jiyao and Ma, Zhiyuan and Wu, Tianhao and Chen, Zeyuan and Dong, Hao},
  booktitle={39th Annual Conference on Neural Information Processing Systems},
  year={2025}
}
```

## License
This work and the dataset are licensed under [CC BY-NC 4.0][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png