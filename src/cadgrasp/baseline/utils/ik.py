import numpy as np
import ikpy.chain
from cadgrasp.baseline.utils.robot_info import UR5_START_QPOS
import transforms3d


class IK:
    def __init__(self) -> None:
        self.arm = ikpy.chain.Chain.from_urdf_file('robot_models/urdf/ur5.urdf', active_links_mask=[False, True, True, True, True, True, True])
        self.arm_full = ikpy.chain.Chain.from_urdf_file('robot_models/urdf/ur5_full.urdf', active_links_mask=[False, True, True, True, True, True, True, False])
    
    def fk(self, joints: np.array = UR5_START_QPOS):
        """
            Forward kinematics function
            joints: joint angles, (6,)
            
            return rotation matrix and translation vector, (3, 3), (3,)
        """
        
        fk = self.arm_full.forward_kinematics(np.concatenate(([0], joints, [0])).tolist())
        fk = np.array(fk).reshape(4, 4)
        return fk[:3, :3], fk[:3, 3]

    def ik(self, trans: np.array, rot: np.array, joints: np.array = UR5_START_QPOS):
        """
            Inverse kinematics function
            trans: translation in camera frame, (3,)
            rot: rotation in camera frame, (3, 3)
            joints: current joints, (6,)
            
            return joint angles, (6,)
        """
        aug_joints = np.concatenate(([0], joints))
        
        ik = self.arm_full.inverse_kinematics(target_position=trans, 
                                        target_orientation=rot, 
                                        orientation_mode='all',
                                        initial_position=aug_joints.tolist()+[0],)
        
        fk2 = self.arm_full.forward_kinematics(ik)
        if not ((np.diag(fk2[:3, :3]@rot.T).sum()-1)/2 > 0.99 and np.linalg.norm(fk2[:3, 3] - trans) < 0.005):
            print(f'unreachable trans {trans}, rot {rot}')
            raise ValueError
        
        return ik[1:7]