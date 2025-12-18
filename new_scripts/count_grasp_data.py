import os,numpy as np

p = '/data/zhiyuanma/DexGraspNet2/IBSGrasp/data'
scenes = os.listdir(p)

for scene in scenes:
    data = np.load(os.path.join(p, scene, 'success_grasps.npz'),allow_pickle=True)['arr_0'].tolist()
    gps = data['translation']
    unique_indices = np.unique(gps, axis=0, return_index=False, return_counts=False)
    unique_count = len(unique_indices)
    print(f"场景{scene} 总共有 {gps.shape[0]} 个子数组，其中唯一子数组数量为 {unique_count}")
