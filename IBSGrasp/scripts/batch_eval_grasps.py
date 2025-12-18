import os, sys
sys.path.append('.')
from IBSGrasp.scripts.evaluate_grasps import main as eval
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
        eval(cfg)
        logger.info(f"Done for {cfg.scene_id}")
    except Exception as e:
        logger.add("DatasetProcessing/bacth_validation_errors.log")
        logger.error(e)

@hydra.main(version_base="v1.2", config_path='../conf', config_name='eval')
def main(cfg: DictConfig):
    scene_ids = range(100)
    
    cfg_list = []
    for scene_id in scene_ids:
        cfg.scene_id = "scene_"+str(scene_id).zfill(4)
        cfg_list.append(deepcopy(cfg))
    
    with Pool(len(gpu_list)) as p:
        result = list(tqdm(
            p.imap_unordered(single_main, cfg_list, chunksize=1),
            total=len(cfg_list),
            desc='Validating scenes（实时进度）'
        ))

if __name__ == "__main__":
    main()
    