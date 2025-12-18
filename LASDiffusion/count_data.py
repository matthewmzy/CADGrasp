import os
import numpy as np

from tqdm import tqdm

scene_ids = range(100)
path = "../IBSGrasp/outputdata"
scene_max_data = 50000

count = 0

for scene_id in tqdm(scene_ids):
    scene_name = 'scene_' + str(scene_id).zfill(4)
    ibs_path = os.path.join(path, 'ibs', scene_name + '.npy')
    w2h_trans_path = os.path.join(path, 'w2h_trans', scene_name + '.npy')
    ibs_data = np.load(ibs_path)
    num_samples = len(ibs_data)
    print(num_samples)
    num_samples = min(num_samples, scene_max_data)
    count+=num_samples

print(count)