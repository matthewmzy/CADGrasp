python src/eval/predict_dexterous_all.py \
    --ckpt data/DexGraspNet2.0-ckpts/SCALING_GRASP_OURS_1000/ckpt/ckpt_50000.pth \
    --dataset graspnet \
    --scene_id_start 9000 \
    --scene_id_end 9900 \
    --overwrite 0 \
    --strategy my_top_5 \
    --exp_name main_result_random

python src/eval/evaluate_dexterous_all.py \
    --dataset graspnet \
    --split random \
    --ckpt_path_list data/DexGraspNet2.0-ckpts/SCALING_GRASP_OURS_1000/ckpt/ckpt_50000.pth \
    --overwrite 0 \
    --exp_name main_result_random