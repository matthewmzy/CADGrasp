import os,numpy as np
p = '/data/zhiyuanma/DexGraspNet2/IBSGrasp/fpsTargetDataNew/ibs/scene_0013.npy'
arr = np.load(p)
B = arr.shape[0]
flattened = arr.reshape(B, -1)  # 形状: (B, 40*40*40*3)

# 找到唯一子数组及其索引和计数
unique_indices = np.unique(flattened, axis=0, return_index=False, return_counts=False)

unique_count = len(unique_indices)
print(f"总共有 {B} 个子数组，其中唯一子数组数量为 {unique_count}")
# print(f"每个唯一子数组的出现次数: {counts}")
# print(f"唯一子数组的原始索引: {unique_indices}")