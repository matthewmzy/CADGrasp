# python IBSGrasp/scripts/batch_cal_ibs.py
# /root/anaconda3/envs/Dex2/bin/python /DATA/disk0/zhiyuanma/DexGraspNet2/IBSGrasp/scripts/annotate_ibs_for_view.py

# cd /DATA/disk0/zhiyuanma/DexGraspNet2/LASDiffusion
# conda init
# conda activate cedex

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python LASDiffusion/train.py \
    --verbose False \
    --batch_size 4 \
    --ibs_load_per_scene 16 \
    --lr 6e-5 \
    --model_mode diffusion \
    --name LEAP_dif \
    --continue_training True

