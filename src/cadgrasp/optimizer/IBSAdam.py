"""
IBS-based Adam Optimizer for Dexterous Hand Grasp Optimization.

This module implements an optimization framework that uses Interaction Bisector Surface (IBS)
to guide hand pose optimization for grasping.
"""

import torch
import torch.nn.functional as F
import numpy as np
from termcolor import cprint

from cadgrasp.optimizer.HandModel import get_handmodel
from cadgrasp.optimizer.rot6d import add_transform_perturbation


class IBSAdam:
    """
    Adam-based optimizer for hand grasp pose optimization using IBS guidance.
    
    The optimizer minimizes a combination of energy terms:
    - Contact matching energy: Align hand surface with IBS contact points
    - Joint limits energy: Keep joints within valid ranges
    - Self-penetration energy: Prevent hand self-collision
    - Penetration energy: Prevent hand-object penetration based on IBS normals
    """
    
    # Default hyperparameters
    DEFAULT_IBS_SIZE = 1024
    DEFAULT_CONTACT_SIZE = 80
    DEFAULT_THUMB_CONTACT_SIZE = 20
    DEFAULT_JOINT_ANGLES = [0, 0.3, 0.3, 0, 0, 0.3, 0.3, 0, 0, 0.3, 0.3, 0., 1.3, 0, 0, 0]
    
    # Energy weights
    WEIGHT_JOINT = 5.0
    WEIGHT_CONTACT_FINGER_TO_IBS = 80.0
    WEIGHT_CONTACT_IBS_TO_FINGER = 2.0
    WEIGHT_CONTACT_THUMB_TO_IBS = 100.0
    WEIGHT_CONTACT_IBS_TO_THUMB = 0.0
    WEIGHT_PENETRATION = 1000.0
    WEIGHT_TRANSFORM = 0.0

    def __init__(
        self,
        hand_name: str,
        parallel_num: int = 10,
        running_name: str = None,
        device: str = None,
        verbose_energy: bool = False,
        ik_step: int = 0
    ):
        """
        Initialize the IBS Adam optimizer.
        
        Args:
            hand_name: Name of the hand model to use
            parallel_num: Number of parallel optimization particles per IBS
            running_name: Name for this optimization run (for logging)
            device: Device to run on ('cuda' or 'cpu')
            verbose_energy: Whether to print energy values during optimization
            ik_step: Number of initial IK steps before full optimization
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.hand_name = hand_name
        self.parallel_num = parallel_num
        self.running_name = running_name
        self.verbose_energy = verbose_energy
        self.ik_step = ik_step
        
        # State variables (initialized in reset())
        self.handmodel = None
        self.num_particles = 0
        self.global_step = 0
        self.q_current = None
        self.initial_pose = None
        self.energy = None
        
        # IBS point clouds
        self.ibs_pcd = None          # (B, N, 6) - positions + normals
        self.cont_pcd = None         # (B, M, 6) - contact points
        self.thumb_cont_pcd = None   # (B, K, 6) - thumb contact points
        
        # Joint limits
        self.q_joint_lower = None
        self.q_joint_upper = None
        
        # Optimizer state
        self.optimizer = None
        self.scheduler = None
        self.test_translation = None
        self.test_rotation = None
        self.test_joint = None
        self.translation_weight = 1.0
        self.rotation_weight = 1.0
        self.joint_weight = 1.0

    def reset(
        self,
        ibs_triplets: list,
        running_name: str,
        cone_viz_num: int = 0,
        cone_mu: float = 1.0,
        filt_or_not: bool = True,
        verbose: bool = False
    ):
        """
        Reset optimizer with new IBS data.
        
        Args:
            ibs_triplets: List of (ibs_occupancy, ibs_contact, ibs_thumb_contact) tuples
            running_name: Name for this optimization run
            cone_viz_num: Number of friction cones to visualize (unused, kept for API compatibility)
            cone_mu: Friction coefficient (unused, kept for API compatibility)
            filt_or_not: Whether to filter IBS (unused, kept for API compatibility)
            verbose: Whether to print verbose output
        """
        self.running_name = running_name
        self.global_step = 0
        
        # Process IBS triplets
        ibs_list, cont_list, thumb_cont_list = self._process_ibs_triplets(ibs_triplets)
        
        self.num_particles = len(ibs_triplets) * self.parallel_num
        
        # Initialize hand model
        self.handmodel = get_handmodel(
            self.hand_name, self.num_particles, self.device, hand_scale=1.
        )
        self.q_joint_lower = self.handmodel.revolute_tendon_q_lower.detach()
        self.q_joint_upper = self.handmodel.revolute_tendon_q_upper.detach()
        
        # Stack point clouds
        self.ibs_pcd = torch.stack(ibs_list, dim=0)
        self.cont_pcd = torch.stack(cont_list, dim=0)
        self.thumb_cont_pcd = torch.stack(thumb_cont_list, dim=0)
        
        # Initialize pose
        self.initial_pose = self._compute_initial_pose()
        self.q_current = self.initial_pose.clone()

    def _process_ibs_triplets(self, ibs_triplets: list) -> tuple:
        """
        Process IBS triplets into point cloud tensors.
        
        Args:
            ibs_triplets: List of (occupancy, contact, thumb_contact) numpy arrays
            
        Returns:
            Tuple of (ibs_list, cont_list, thumb_cont_list) tensors
        """
        ibs_list, cont_list, thumb_cont_list = [], [], []
        
        for ibs_occu, ibs_cont, ibs_thumb_cont in ibs_triplets:
            # Subsample occupancy points
            ibs_occu = self._subsample_points(ibs_occu, self.DEFAULT_IBS_SIZE)
            
            # Process contact points
            if ibs_cont.shape[0] == 0:
                cprint("[WARNING] No contact points found, using IBS points as fallback", 'red')
                ibs_cont = ibs_occu[:self.DEFAULT_CONTACT_SIZE]
            else:
                ibs_cont = self._subsample_points(ibs_cont, self.DEFAULT_CONTACT_SIZE)
            
            # Process thumb contact points
            if ibs_thumb_cont.shape[0] == 0:
                cprint("[WARNING] No thumb contact points found, using contact points as fallback", 'red')
                ibs_thumb_cont = ibs_cont[:self.DEFAULT_THUMB_CONTACT_SIZE]
            else:
                ibs_thumb_cont = self._subsample_points(ibs_thumb_cont, self.DEFAULT_THUMB_CONTACT_SIZE)
            
            # Convert to tensors and replicate for parallel particles
            ibs_tensor = torch.from_numpy(ibs_occu).float().to(self.device)
            cont_tensor = torch.from_numpy(ibs_cont).float().to(self.device)
            thumb_tensor = torch.from_numpy(ibs_thumb_cont).float().to(self.device)
            
            ibs_list.extend([ibs_tensor] * self.parallel_num)
            cont_list.extend([cont_tensor] * self.parallel_num)
            thumb_cont_list.extend([thumb_tensor] * self.parallel_num)
        
        return ibs_list, cont_list, thumb_cont_list

    @staticmethod
    def _subsample_points(points: np.ndarray, target_size: int) -> np.ndarray:
        """Randomly subsample points to target size."""
        n_points = points.shape[0]
        indices = np.random.choice(n_points, target_size, replace=(n_points < target_size))
        return points[indices]

    def _compute_initial_pose(self) -> torch.Tensor:
        """Compute initial hand pose from w2h transformation."""
        # Get inverse transformation and add offset
        transform = torch.inverse(self.handmodel.w2h_trans_meta).to(self.device)
        
        # Move hand back along z-axis
        moveback = torch.eye(4, device=self.device)
        moveback[2, 3] = 0.2
        transform = torch.matmul(transform, moveback)
        
        # Add random perturbation
        transform = add_transform_perturbation(transform, max_translation=0.03, max_rotation=10)
        
        # Extract translation and rotation (6D representation)
        translation = transform[:, :3, 3]
        rotation = transform[:, :3, :3].transpose(1, 2).reshape(self.num_particles, -1)[:, :6]
        
        # Default joint angles
        joint_angles = torch.tensor(
            [self.DEFAULT_JOINT_ANGLES], 
            device=self.device
        ).repeat(self.num_particles, 1)
        
        return torch.cat([translation, rotation, joint_angles], dim=1)

    def set_opt_weight(
        self,
        w_trans: float,
        w_rot: float,
        w_joint: float,
        learning_rate: float,
        decay_every: int,
        lr_decay: float
    ):
        """
        Set optimization weights and initialize optimizer.
        
        Args:
            w_trans: Weight for translation parameters
            w_rot: Weight for rotation parameters
            w_joint: Weight for joint parameters
            learning_rate: Initial learning rate
            decay_every: Steps between learning rate decay
            lr_decay: Learning rate decay factor
        """
        self.translation_weight = w_trans
        self.rotation_weight = w_rot
        self.joint_weight = w_joint
        
        # Create weighted parameter tensors
        self.test_translation = self.q_current[:, :3].detach().clone() * w_trans
        self.test_rotation = self.q_current[:, 3:9].detach().clone() * w_rot
        self.test_joint = self.q_current[:, 9:].detach().clone() * w_joint
        
        # Enable gradients
        self.test_translation.requires_grad_(True)
        self.test_rotation.requires_grad_(True)
        self.test_joint.requires_grad_(True)
        
        # Setup optimizer and scheduler
        params = [self.test_translation, self.test_rotation, self.test_joint]
        self.optimizer = torch.optim.AdamW(params, lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=decay_every, gamma=lr_decay
        )

    # ==================== Energy Functions ====================

    def contact_match_energy(self) -> tuple:
        """
        Compute contact matching energy between hand and IBS contact points.
        
        Returns:
            Tuple of (finger_to_ibs, ibs_to_finger, thumb_to_ibs, ibs_to_thumb) energies
        """
        # Get palmar surface points
        _, _, finger_pts, _, thumb_pts, _ = self.handmodel.get_surface_points_and_normals(palmar=True)
        
        # Finger contact matching
        finger_dist = torch.cdist(self.cont_pcd[:, :, :3], finger_pts)
        e_finger_to_ibs = finger_dist.min(dim=2)[0].mean(dim=1)
        e_ibs_to_finger = finger_dist.min(dim=1)[0].mean(dim=1)
        
        # Thumb contact matching
        thumb_dist = torch.cdist(self.thumb_cont_pcd[:, :, :3], thumb_pts)
        e_thumb_to_ibs = thumb_dist.min(dim=2)[0].mean(dim=1)
        e_ibs_to_thumb = thumb_dist.min(dim=1)[0].mean(dim=1)
        
        return e_finger_to_ibs, e_ibs_to_finger, e_thumb_to_ibs, e_ibs_to_thumb

    def joint_limits_energy(self) -> torch.Tensor:
        """Compute energy penalizing joint limit violations."""
        joint_angles = self.q_current[:, 9:]
        over_upper = F.relu(joint_angles - self.q_joint_upper).sum(dim=1)
        under_lower = F.relu(self.q_joint_lower - joint_angles).sum(dim=1)
        return over_upper + under_lower

    def self_penetration_energy(self) -> torch.Tensor:
        """Compute energy penalizing hand self-penetration."""
        return self.handmodel.get_self_penetration()

    def penetration_energy_improved(self) -> torch.Tensor:
        """
        Compute penetration energy using IBS normals.
        
        Penalizes hand surface points that appear on the wrong side of IBS 
        (opposite to the normal direction), indicating penetration.
        """
        # Get hand surface points and IBS data
        surface_pts, _ = self.handmodel.get_surface_points_and_normals(palmar=False)
        ibs_pos = self.ibs_pcd[:, :, :3]
        ibs_normal = self.ibs_pcd[:, :, 3:]
        
        # Find nearest IBS point for each surface point
        dist_mat = torch.cdist(surface_pts, ibs_pos)
        _, min_idx = dist_mat.min(dim=2)
        
        # Gather nearest IBS points and normals
        batch_idx = torch.arange(self.num_particles, device=self.device).unsqueeze(1)
        nearest_pos = ibs_pos[batch_idx, min_idx]
        nearest_normal = ibs_normal[batch_idx, min_idx]
        
        # Compute vector from IBS to surface point
        vec_to_surface = surface_pts - nearest_pos
        
        # Check if on wrong side (negative dot product = penetration)
        dot_product = (vec_to_surface * nearest_normal).sum(dim=2)
        
        # Penalize only penetrating points
        penalty = F.relu(-dot_product)
        
        return penalty.sum(dim=1) / surface_pts.shape[1]

    def transform_regularization_energy(self) -> torch.Tensor:
        """Compute energy penalizing deviation from initial pose."""
        current_transform = self.q_current[:, :9]
        initial_transform = self.initial_pose[:, :9]
        return torch.norm(current_transform - initial_transform, dim=1)

    def compute_energy(self, energy_dict: dict = None) -> torch.Tensor:
        """
        Compute total energy as weighted sum of all energy terms.
        
        Args:
            energy_dict: Dictionary to store individual energy values
            
        Returns:
            Total energy tensor of shape (num_particles,)
        """
        # Compute individual energies
        E_joint = self.joint_limits_energy() * self.WEIGHT_JOINT
        E_spen = self.self_penetration_energy()
        E_cont_1, E_cont_2, E_cont_3, E_cont_4 = self.contact_match_energy()
        E_pen = self.penetration_energy_improved() * self.WEIGHT_PENETRATION
        E_trans = self.transform_regularization_energy() * self.WEIGHT_TRANSFORM
        
        # Apply contact weights
        E_cont_1 = E_cont_1 * self.WEIGHT_CONTACT_FINGER_TO_IBS
        E_cont_2 = E_cont_2 * self.WEIGHT_CONTACT_IBS_TO_FINGER
        E_cont_3 = E_cont_3 * self.WEIGHT_CONTACT_THUMB_TO_IBS
        E_cont_4 = E_cont_4 * self.WEIGHT_CONTACT_IBS_TO_THUMB
        
        # Total energy
        total = E_joint + E_spen + E_cont_1 + E_cont_2 + E_cont_3 + E_cont_4 + E_pen + E_trans
        
        # Log individual energies
        if energy_dict is not None:
            with torch.no_grad():
                energy_dict['E_joint'].append(E_joint)
                energy_dict['E_spen'].append(E_spen)
                energy_dict['E_cont_1'].append(E_cont_1)
                energy_dict['E_cont_2'].append(E_cont_2)
                energy_dict['E_cont_3'].append(E_cont_3)
                energy_dict['E_cont_4'].append(E_cont_4)
                energy_dict['E_pen'].append(E_pen)
                energy_dict['E_trans'].append(E_trans)
        
        self.energy = total
        return total

    # ==================== Optimization Step ====================

    def step(self, energy_dict: dict = None):
        """
        Perform one optimization step.
        
        Args:
            energy_dict: Dictionary to store energy values for logging
        """
        self.optimizer.zero_grad()
        
        # Reconstruct q_current from weighted parameters
        self.q_current = torch.cat([
            self.test_translation / self.translation_weight,
            self.test_rotation / self.rotation_weight,
            self.test_joint / self.joint_weight
        ], dim=1)
        
        # Update kinematics
        self.handmodel.update_kinematics(q=self.q_current)
        
        # Compute energy and backprop
        energy = self.compute_energy(energy_dict)
        energy.mean().backward()
        
        # Update parameters
        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1

    def get_opt_q(self) -> torch.Tensor:
        """Get current optimized pose parameters."""
        return self.q_current.detach()
