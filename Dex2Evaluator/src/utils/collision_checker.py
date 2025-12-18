import torch

from src.utils.robot_model import RobotModel


class CollisionChecker:
    """
    class for checking collision between grasps and scene point cloud
    """
    
    def __init__(
        self, 
        config: dict, 
        device: torch.device, 
    ):
        """
        initialize the class
        
        Args:
        - config: dict, config of the simulation evaluator
        - device: torch.device, device
        """
        self._config: dict = config
        self._device: torch.device = torch.device(device)
        
        # robot model
        self._robot_model = RobotModel(
            urdf_path=self._config['urdf_path'], 
            meta_path=self._config['meta_path'], 
        )
    
    def check_collision_batch(
        self, 
        grasps: dict,
        scene_point_cloud: torch.Tensor,
        with_table: bool = True,
    ):
        """
        check collision between grasps and scene point cloud
        
        Args:
        - grasps: dict[str, torch.Tensor], grasps
        - scene_point_cloud: torch.Tensor[n_points, 3], scene point cloud
        - with_table: bool, whether to check collision with table
        
        Returns:
        - scene_pen_distance: torch.Tensor[batch_size], scene penetration depth
        - table_pen_distance: torch.Tensor[batch_size], table penetration depth
        """
        
        batch_size = len(grasps['translation'])
        
        # forward kinematics
        local_translations, local_rotations = self._robot_model.forward_kinematics(
            { joint_name: grasps[joint_name] for joint_name in self._robot_model.joint_names })
        global_translations = grasps['translation']
        global_rotations = grasps['rotation']
        
        # compute scene penetration
        scene_pen_distance = self._robot_model.cal_distance(
            local_translations=local_translations,
            local_rotations=local_rotations,
            global_translation=global_translations,
            global_rotation=global_rotations,
            x=scene_point_cloud.unsqueeze(0).repeat(batch_size, 1, 1),
            dilation_pen=0.,
        ).max(dim=1).values
        
        if not with_table:
            return scene_pen_distance
        
        # compute table penetration
        plane_parameters = torch.tensor([[0, 0, 1, 0]], 
            dtype=torch.float, device=self._device).repeat(batch_size, 1)
        table_pen_distance = self._robot_model.cal_dis_plane(
            local_translations=local_translations,
            local_rotations=local_rotations,
            global_translation=global_translations,
            global_rotation=global_rotations,
            p=plane_parameters,
            dilation_tpen=0.,
        )
        
        return scene_pen_distance, table_pen_distance    
