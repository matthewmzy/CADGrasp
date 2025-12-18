# python src/eval/predict_dexterous_all.py \
#     --ckpt data/DexGraspNet2.0-ckpts/SCALING_GRASP_OURS_1000/ckpt/ckpt_50000.pth \
#     --dataset graspnet \
#     --scene_id_start 9000 \
#     --scene_id_end 9900 \
#     --overwrite 0 \
#     --strategy my_top_5 \
#     --exp_name cross_embodiment_allegro \
#     --ibs_exp_name main_result_random \
#     --test_hand allegro

# python src/eval/predict_dexterous_all.py \
#     --ckpt data/DexGraspNet2.0-ckpts/SCALING_GRASP_OURS_1000/ckpt/ckpt_50000.pth \
#     --dataset graspnet \
#     --scene_id_start 200 \
#     --scene_id_end 380 \
#     --overwrite 0 \
#     --strategy my_top_5 \
#     --exp_name cross_embodiment_allegro \
#     --ibs_exp_name main_results_loose \
#     --test_hand allegro

# python src/eval/evaluate_dexterous_all.py \
#     --dataset graspnet \
#     --split random \
#     --ckpt_path_list data/DexGraspNet2.0-ckpts/SCALING_GRASP_OURS_1000/ckpt/ckpt_50000.pth \
#     --overwrite 0 \
#     --exp_name cross_embodiment_allegro \
#     --robot_name allegro

# python src/eval/evaluate_dexterous_all.py \
#     --dataset graspnet \
#     --split loose \
#     --ckpt_path_list data/DexGraspNet2.0-ckpts/SCALING_GRASP_OURS_1000/ckpt/ckpt_50000.pth \
#     --overwrite 0 \
#     --exp_name cross_embodiment_allegro \
#     --robot_name allegro

# python src/eval/print_dexterous_results.py \
#     --ckpt_path data/DexGraspNet2.0-ckpts/SCALING_GRASP_OURS_1000/ckpt/ckpt_50000.pth \
#     --exp_name cross_embodiment_allegro \

python src/eval/predict_dexterous_all.py \
    --ckpt data/DexGraspNet2.0-ckpts/SCALING_GRASP_OURS_1000/ckpt/ckpt_50000.pth \
    --dataset acronym \
    --overwrite 0 \
    --strategy my_top_5 \
    --exp_name cross_embodiment_allegro \
    --ibs_exp_name main_result_acronym

python src/eval/evaluate_dexterous_all.py \
    --dataset acronym \
    --split dense \
    --ckpt_path_list data/DexGraspNet2.0-ckpts/SCALING_GRASP_OURS_1000/ckpt/ckpt_50000.pth \
    --overwrite 0 \
    --exp_name cross_embodiment_allegro \
    --robot_name allegro

python src/eval/evaluate_dexterous_all.py \
    --dataset acronym \
    --split random \
    --ckpt_path_list data/DexGraspNet2.0-ckpts/SCALING_GRASP_OURS_1000/ckpt/ckpt_50000.pth \
    --overwrite 0 \
    --exp_name cross_embodiment_allegro \
    --robot_name allegro

python src/eval/evaluate_dexterous_all.py \
    --dataset acronym \
    --split loose \
    --ckpt_path_list data/DexGraspNet2.0-ckpts/SCALING_GRASP_OURS_1000/ckpt/ckpt_50000.pth \
    --overwrite 0 \
    --exp_name cross_embodiment_allegro \
    --robot_name allegro

python src/eval/print_dexterous_results.py \
    --ckpt_path data/DexGraspNet2.0-ckpts/SCALING_GRASP_OURS_1000/ckpt/ckpt_50000.pth \
    --exp_name cross_embodiment_allegro \