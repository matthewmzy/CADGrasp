CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python LASDiffusion/train.py \
    --verbose False \
    --batch_size 4 \
    --ibs_load_per_scene 16 \
    --lr 6e-5 \
    --model_mode diffusion \
    --name LEAP_dif_100 \
    --continue_training False \
    --scaling 100
