from typing import Union
from src.utils.data_evaluator.simulation_evaluator import SimulationEvaluator
import torch


class DataEvaluator:
    """
    base class for data evaluator
    """
    
    def __init__(
        self, 
        config: dict, 
        device: Union[str, torch.device],
    ):
        """
        initialize the class
        
        Args:
        - config: dict, config of the data evaluator
        - device: Union[str, torch.device], device
        """
        self._config: dict = config
        self._device: torch.device = torch.device(device)

    def evaluate_data(
        self, 
        object_pose_dict: dict,
        grasps: dict,
    ):
        """
        evaluate a batch of grasps on one cluttered scene
        
        Args:
        - object_pose_dict: dict[str, np.ndarray[4, 4]], object code -> pose
        - grasps: dict[str, np.ndarray], grasps, format: {
            'translation': np.ndarray[batch_size, 3], translations,
            'rotation': np.ndarray[batch_size, 3, 3], rotations,
            'jointxxx': np.ndarray[batch_size], joint values,
            ...
        }
        
        Returns:
        - successes: np.ndarray[batch_size, np.bool], successes
        """
        raise NotImplementedError()

def get_evaluator(evaluator_type: str):
    """
    get evaluator class
    
    Args:
    - evaluator_type: str, type of the data evaluator
    Returns:
    - type[DataEvaluator], data evaluator class
    """
    if evaluator_type == 'SimulationEvaluator':
        return SimulationEvaluator
    else:
        raise ValueError(f'invalid evaluator type: {evaluator_type}')
