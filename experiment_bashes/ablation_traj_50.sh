python src/eval/predict_dexterous_all.py \
    --ckpt data/DexGraspNet2.0-ckpts/SCALING_GRASP_OURS_1000/ckpt/ckpt_50000.pth \
    --dataset graspnet \
    --scene_id_start 100 \
    --scene_id_end 190 \
    --overwrite 0 \
    --strategy my_top_5 \
    --ibs_exp_name main_result \
    --exp_name ablation_traj_para_50

python src/eval/evaluate_dexterous_all.py \
    --dataset graspnet \
    --split dense \
    --ckpt_path_list data/DexGraspNet2.0-ckpts/SCALING_GRASP_OURS_1000/ckpt/ckpt_50000.pth \
    --overwrite 0 \
    --exp_name ablation_traj_para_50

python src/eval/print_dexterous_results.py \
    --ckpt_path data/DexGraspNet2.0-ckpts/SCALING_GRASP_OURS_1000/ckpt/ckpt_50000.pth \
    --exp_name ablation_traj_para_50
