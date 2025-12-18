import os, sys
import numpy as np
import torch
from tqdm import tqdm
sys.path.append('.')
from IBSGrasp.utils.transforms import batch_transform_points, transform_points, pc_plotly, show
import os, sys
sys.path.append('.')
from IBSGrasp.scripts.calculate_ibs import calculate_IBS
import numpy as np
import hydra
import signal
from copy import deepcopy
from tqdm import tqdm
from omegaconf import DictConfig
from loguru import logger
from torch.multiprocessing import set_start_method, Pool, current_process

try:
    set_start_method('spawn')
except RuntimeError:
    pass

fps_data_path = 'IBSGrasp/ibsdata'
scene_path = 'data/scenes'
valid_ids_path = 'IBSGrasp/scene_valid_ids' 
scene_ids = range(100)
view_ids = range(256)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
np.seterr(all='raise') 
gpu_list = [str(i) for i in range(8)]
enable_vis = False

def handle_timeout(signum, frame):
    raise RuntimeError("Timeout")


def single_main(scene_id):
    try:
        # signal.signal(signal.SIGALRM, handle_timeout)
        # signal.alarm(60*30)
        worker = current_process()._identity[0]
        # print(f"worker {worker} start")
        device = f"cuda:{gpu_list[(worker - 1)%len(gpu_list)]}"
        scene_name = 'scene_'+str(scene_id).zfill(4)
        ibs_path = os.path.join(fps_data_path, 'ibs', scene_name + '.npy')
        w2h_trans_path = os.path.join(fps_data_path, 'w2h_trans', scene_name + '.npy')
        ibs = np.load(ibs_path)
        w2h_trans = np.load(w2h_trans_path)
        grasppoints = np.linalg.inv(w2h_trans)[:,:3,3]
        grasppoints = torch.from_numpy(grasppoints).float().to(device=device)
        os.makedirs(os.path.join(valid_ids_path, scene_name), exist_ok=True)
        for view_id in tqdm(view_ids):
            vp = os.path.join(valid_ids_path, scene_name, 'view_'+str(view_id).zfill(4)+'.npy')
            if os.path.exists(vp):
                continue
            scene_data = np.load(os.path.join(scene_path, scene_name, 'realsense', 'network_input.npz'))
            scene_pc_c = scene_data['pc'][view_id]  # (N, 3)
            scene_pc_c = torch.from_numpy(scene_pc_c).float().to(device=device)
            c2w_trans = scene_data['extrinsics'][view_id]
            c2w_trans = torch.from_numpy(c2w_trans).float().to(device=device)
            scene_pc_w = transform_points(scene_pc_c, c2w_trans)  # (N, 3)， 世界坐标系
            if enable_vis and view_id==0:
                show([pc_plotly(scene_pc_w), pc_plotly(grasppoints,color='red')])
            dist = torch.cdist(grasppoints, scene_pc_w)
            min_dis = torch.min(dist, dim=1)[0]
            valid_ids = (min_dis<0.01)
            if enable_vis and view_id==0:
                show([pc_plotly(scene_pc_w), pc_plotly(grasppoints[valid_ids],color='red')])
            valid_ids = valid_ids.cpu().numpy()
            np.save(vp, valid_ids)
        logger.info(f"Done for {scene_id}")
    except Exception as e:
        logger.add("DatasetProcessing/bacth_validation_errors.log")
        logger.error(e)


@hydra.main(version_base="v1.2", config_path='../conf', config_name='ibs')
def main(cfg: DictConfig):
    pose_data_path = cfg.pose_data_path
    total_scene_nums = len(os.listdir(pose_data_path))

    
    
    with Pool(len(gpu_list)) as p:
        result = list(tqdm(
            p.imap_unordered(single_main, scene_ids, chunksize=1),
            total=len(scene_ids),
            desc='Validating scenes（实时进度）'
        ))

if __name__ == "__main__":
    main()


    


