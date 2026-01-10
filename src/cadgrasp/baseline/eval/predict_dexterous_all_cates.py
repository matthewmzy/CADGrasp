import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str)
args = parser.parse_args()

os.system(f'python src/eval/predict_dexterous_all.py --ckpt {args.ckpt} --dataset graspnet --scene_id_start 100 --scene_id_end 190')
os.system(f'python src/eval/predict_dexterous_all.py --ckpt {args.ckpt} --dataset graspnet --scene_id_start 200 --scene_id_end 380')
os.system(f'python src/eval/predict_dexterous_all.py --ckpt {args.ckpt} --dataset graspnet --scene_id_start 9000 --scene_id_end 9900')
os.system(f'python src/eval/predict_dexterous_all.py --ckpt {args.ckpt} --dataset acronym')
