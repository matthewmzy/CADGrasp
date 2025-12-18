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

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
np.seterr(all='raise') 
gpu_list = [str(i) for i in range(8)]


def handle_timeout(signum, frame):
    raise RuntimeError("Timeout")


def single_main(cfg: DictConfig):
    try:
        # signal.signal(signal.SIGALRM, handle_timeout)
        # signal.alarm(60*30)
        worker = current_process()._identity[0]
        # print(f"worker {worker} start")
        device = f"cuda:{gpu_list[(worker - 1)%len(gpu_list)]}"
        cfg.device = device
        calculate_IBS(cfg)
        logger.info(f"Done for {cfg.scene_id}")
    except Exception as e:
        logger.add("DatasetProcessing/bacth_validation_errors.log")
        logger.error(e)


@hydra.main(version_base="v1.2", config_path='../conf', config_name='ibs')
def main(cfg: DictConfig):
    pose_data_path = cfg.pose_data_path
    total_scene_nums = len(os.listdir(pose_data_path))

    cfg_list = []
    for i in range(65,total_scene_nums):
        cfg.scene_id = i
        cfg.vis = 0
        cfg_list.append(deepcopy(cfg))
    
    with Pool(len(gpu_list)) as p:
        result = list(tqdm(
            p.imap_unordered(single_main, cfg_list, chunksize=1),
            total=len(cfg_list),
            desc='Validating scenes（实时进度）'
        ))

if __name__ == "__main__":
    main()
    