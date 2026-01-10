import os

for dataset in ['graspnet', 'acronym']:
    for split in ['dense', 'random', 'loose']:
        os.system(f'python src/eval/evaluate_dexterous_all.py --dataset {dataset} --split {split} ')