python src/eval/predict_dexterous_all.py \
    --ckpt data/DexGraspNet2.0-ckpts/SCALING_GRASP_OURS_1000/ckpt/ckpt_50000.pth \
    --dataset graspnet \
    --scene_id_start 100 \
    --scene_id_end 190 \
    --overwrite 0 \
    --strategy my_top_5 \
    --exp_name scaling_100 \
    --ibs_exp_name scaling_100 \
    --las_model_path LASDiffusion/results/LEAP_dif_100/history/epoch_epoch=0100.ckpt \
    --test_hand leap_hand

python src/eval/evaluate_dexterous_all.py \
    --dataset graspnet \
    --split dense \
    --ckpt_path_list data/DexGraspNet2.0-ckpts/SCALING_GRASP_OURS_1000/ckpt/ckpt_50000.pth \
    --overwrite 0 \
    --exp_name scaling_100

python src/eval/print_dexterous_results.py \
    --ckpt_path data/DexGraspNet2.0-ckpts/SCALING_GRASP_OURS_1000/ckpt/ckpt_50000.pth \
    --exp_name scaling_100