"""
IBS (Interaction Bisector Surface) Representation Class.

This module provides a unified class for handling IBS data in the CADGrasp project.
IBS is the set of points equidistant from hand and object surfaces.

IBS has two forms:
1. Point cloud form: For visualization and optimizer input
2. Voxel form: For Diffusion network input/output (40x40x40 grid)

IBS voxel has 3 channels:
- Channel 0 (occupancy): IBS occupancy mask (-1 or 1)
- Channel 1 (contact): Finger contact region (1 if contact)
- Channel 2 (thumb_contact): Thumb contact region (2 if thumb contact)

Example usage:
    # Create from file
    ibs = IBS.from_file('path/to/ibs.npy', 'path/to/w2h_trans.npy')
    
    # Create from voxel
    ibs = IBS.from_voxel(voxel_data, w2h_trans)
    
    # Get point cloud
    pc = ibs.to_point_cloud()
    
    # Get contact points
    contact_pc = ibs.get_contact_points()
"""

import os
import numpy as np
import torch
from typing import Optional, Tuple, Union, List
from dataclasses import dataclass


@dataclass
class IBSConfig:
    """Configuration for IBS computation and representation."""
    bound: float = 0.1              # Spatial bound [-bound, bound]^3
    resolution: float = 0.005       # Voxel resolution
    voxel_size: int = 40            # Voxel grid size (40x40x40)
    delta: float = 0.005            # Threshold for IBS mask
    contact_delta: float = 0.0075   # Threshold for contact detection (delta * 1.5)
    thumb_contact_delta: float = 0.0085  # Threshold for thumb contact (delta * 1.7)
    epsilon: float = 1e-5           # Convergence threshold for iterative refinement
    max_iteration: int = 20         # Max iterations for IBS refinement


class IBS:
    """
    Interaction Bisector Surface (IBS) representation class.
    
    IBS is a key intermediate representation in CADGrasp, defined as the 
    set of points equidistant from hand and object surfaces.
    
    Attributes:
        voxel: (40, 40, 40, 3) tensor - IBS voxel with 3 channels
               [occupancy, contact, thumb_contact]
        w2h_trans: (4, 4) transformation from world to hand coordinate
        config: IBSConfig - configuration parameters
        device: torch device
    """
    
    def __init__(
        self,
        voxel: Optional[Union[np.ndarray, torch.Tensor]] = None,
        w2h_trans: Optional[Union[np.ndarray, torch.Tensor]] = None,
        hand_dis: Optional[Union[np.ndarray, torch.Tensor]] = None,
        config: Optional[IBSConfig] = None,
        device: str = 'cpu'
    ):
        """
        Initialize IBS object.
        
        Args:
            voxel: (40, 40, 40, 3) IBS voxel data
            w2h_trans: (4, 4) world to hand transformation
            hand_dis: (40, 40, 40) distance to hand surface (optional)
            config: IBSConfig object
            device: torch device
        """
        self.config = config or IBSConfig()
        self.device = device
        
        # Initialize voxel
        if voxel is not None:
            if isinstance(voxel, np.ndarray):
                voxel = torch.from_numpy(voxel)
            self.voxel = voxel.to(device)
        else:
            self.voxel = None
        
        # Initialize transformation
        if w2h_trans is not None:
            if isinstance(w2h_trans, np.ndarray):
                w2h_trans = torch.from_numpy(w2h_trans)
            self.w2h_trans = w2h_trans.float().to(device)
        else:
            self.w2h_trans = None
        
        # Optional: hand distance field
        if hand_dis is not None:
            if isinstance(hand_dis, np.ndarray):
                hand_dis = torch.from_numpy(hand_dis)
            self.hand_dis = hand_dis.to(device)
        else:
            self.hand_dis = None
        
        # Cache for point cloud representation
        self._point_cloud_cache = None
        self._contact_points_cache = None
        self._thumb_contact_points_cache = None
    
    # ==================== Factory Methods ====================
    
    @classmethod
    def from_file(
        cls,
        ibs_path: str,
        w2h_trans_path: str,
        hand_dis_path: Optional[str] = None,
        index: int = 0,
        device: str = 'cpu'
    ) -> 'IBS':
        """
        Load IBS from numpy files.
        
        Args:
            ibs_path: Path to IBS voxel file (.npy)
            w2h_trans_path: Path to w2h transformation file (.npy)
            hand_dis_path: Optional path to hand distance file (.npy)
            index: Index of the IBS in the batch
            device: torch device
        
        Returns:
            IBS object
        """
        ibs_data = np.load(ibs_path)
        w2h_data = np.load(w2h_trans_path)
        
        # Handle batched data
        if ibs_data.ndim == 5:  # (N, 40, 40, 40, 3)
            voxel = ibs_data[index]
            w2h_trans = w2h_data[index]
        else:
            voxel = ibs_data
            w2h_trans = w2h_data
        
        hand_dis = None
        if hand_dis_path is not None and os.path.exists(hand_dis_path):
            hand_dis_data = np.load(hand_dis_path)
            if hand_dis_data.ndim == 4:  # (N, 40, 40, 40)
                hand_dis = hand_dis_data[index]
            else:
                hand_dis = hand_dis_data
        
        return cls(voxel=voxel, w2h_trans=w2h_trans, hand_dis=hand_dis, device=device)
    
    @classmethod
    def from_voxel(
        cls,
        voxel: Union[np.ndarray, torch.Tensor],
        w2h_trans: Union[np.ndarray, torch.Tensor],
        device: str = 'cpu'
    ) -> 'IBS':
        """
        Create IBS from voxel data.
        
        Args:
            voxel: (40, 40, 40, 3) IBS voxel
            w2h_trans: (4, 4) transformation matrix
            device: torch device
        
        Returns:
            IBS object
        """
        return cls(voxel=voxel, w2h_trans=w2h_trans, device=device)
    
    @classmethod
    def from_point_cloud(
        cls,
        points: Union[np.ndarray, torch.Tensor],
        contact_mask: Union[np.ndarray, torch.Tensor],
        thumb_contact_mask: Union[np.ndarray, torch.Tensor],
        w2h_trans: Union[np.ndarray, torch.Tensor],
        config: Optional[IBSConfig] = None,
        device: str = 'cpu'
    ) -> 'IBS':
        """
        Create IBS from point cloud by voxelization.
        
        Args:
            points: (N, 3) IBS points in hand coordinate
            contact_mask: (N,) boolean mask for contact points
            thumb_contact_mask: (N,) boolean mask for thumb contact points
            w2h_trans: (4, 4) transformation matrix
            config: IBSConfig object
            device: torch device
        
        Returns:
            IBS object
        """
        config = config or IBSConfig()
        
        if isinstance(points, np.ndarray):
            points = torch.from_numpy(points)
        if isinstance(contact_mask, np.ndarray):
            contact_mask = torch.from_numpy(contact_mask)
        if isinstance(thumb_contact_mask, np.ndarray):
            thumb_contact_mask = torch.from_numpy(thumb_contact_mask)
        
        points = points.to(device)
        contact_mask = contact_mask.to(device)
        thumb_contact_mask = thumb_contact_mask.to(device)
        
        # Voxelize points
        voxel = cls._voxelize_points(
            points, contact_mask, thumb_contact_mask, config, device
        )
        
        return cls(voxel=voxel, w2h_trans=w2h_trans, device=device, config=config)
    
    @staticmethod
    def _voxelize_points(
        points: torch.Tensor,
        contact_mask: torch.Tensor,
        thumb_contact_mask: torch.Tensor,
        config: IBSConfig,
        device: str
    ) -> torch.Tensor:
        """Convert point cloud to voxel representation."""
        bound = config.bound
        resolution = config.resolution
        voxel_size = config.voxel_size
        
        # Convert to voxel indices
        voxel_coords = (points - torch.tensor([-bound, -bound, -bound], device=device)) / resolution
        voxel_coords = torch.floor(voxel_coords).long()
        voxel_coords = torch.clamp(voxel_coords, 0, voxel_size - 1)
        
        # Create voxel grids
        occupancy = torch.zeros((voxel_size, voxel_size, voxel_size), dtype=torch.bool, device=device)
        contact = torch.zeros((voxel_size, voxel_size, voxel_size), dtype=torch.bool, device=device)
        thumb_contact = torch.zeros((voxel_size, voxel_size, voxel_size), dtype=torch.bool, device=device)
        
        # Fill voxels
        occupancy[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]] = True
        
        contact_coords = voxel_coords[contact_mask]
        if len(contact_coords) > 0:
            contact[contact_coords[:, 0], contact_coords[:, 1], contact_coords[:, 2]] = True
        
        thumb_coords = voxel_coords[thumb_contact_mask]
        if len(thumb_coords) > 0:
            thumb_contact[thumb_coords[:, 0], thumb_coords[:, 1], thumb_coords[:, 2]] = True
        
        # Stack channels
        voxel = torch.stack([occupancy, contact, thumb_contact], dim=-1)
        
        return voxel
    
    # ==================== Conversion Methods ====================
    
    def to_point_cloud(self, use_cache: bool = True) -> torch.Tensor:
        """
        Convert IBS voxel to point cloud representation.
        
        Args:
            use_cache: Whether to use cached result
        
        Returns:
            (N, 3) point cloud in hand coordinate
        """
        if use_cache and self._point_cloud_cache is not None:
            return self._point_cloud_cache
        
        if self.voxel is None:
            raise ValueError("Voxel data not initialized")
        
        # Get occupancy mask
        occupancy = self.get_occupancy_mask()
        
        # Get voxel coordinates
        indices = torch.nonzero(occupancy, as_tuple=False)  # (N, 3)
        
        # Convert to spatial coordinates
        bound = self.config.bound
        resolution = self.config.resolution
        
        points = indices.float() * resolution + torch.tensor(
            [-bound + resolution/2, -bound + resolution/2, -bound + resolution/2],
            device=self.device
        )
        
        if use_cache:
            self._point_cloud_cache = points
        
        return points
    
    def to_world_points(self) -> torch.Tensor:
        """
        Convert IBS points to world coordinate.
        
        Returns:
            (N, 3) point cloud in world coordinate
        """
        if self.w2h_trans is None:
            raise ValueError("World-to-hand transformation not set")
        
        points_hand = self.to_point_cloud()
        
        # Transform from hand to world
        h2w_trans = torch.inverse(self.w2h_trans)
        
        # Apply transformation
        points_homo = torch.cat([
            points_hand,
            torch.ones((points_hand.shape[0], 1), device=self.device)
        ], dim=1)
        
        points_world = (points_homo @ h2w_trans.T)[:, :3]
        
        return points_world
    
    def to_network_input(self) -> torch.Tensor:
        """
        Convert to network input format (B, C, D, H, W).
        
        For Diffusion network: output is (2, 40, 40, 40)
        - Channel 0: occupancy + contact (combined)
        - Channel 1: thumb_contact
        
        Returns:
            (2, 40, 40, 40) tensor
        """
        if self.voxel is None:
            raise ValueError("Voxel data not initialized")
        
        # Get channels
        occupancy = self.voxel[..., 0].float()
        contact = self.voxel[..., 1].float()
        thumb_contact = self.voxel[..., 2].float()
        
        # Network format: occupancy (-1 for empty, 1 for occupied, 2 for contact)
        # Channel 0: occupancy + contact
        ch0 = torch.where(occupancy > 0, 1.0, -1.0)
        ch0 = torch.where(contact > 0, 2.0, ch0)
        
        # Channel 1: thumb contact
        ch1 = thumb_contact * 2.0
        
        return torch.stack([ch0, ch1], dim=0)  # (2, 40, 40, 40)
    
    # ==================== Access Methods ====================
    
    def get_occupancy_mask(self) -> torch.Tensor:
        """Get occupancy mask (40, 40, 40)."""
        if self.voxel is None:
            raise ValueError("Voxel data not initialized")
        return self.voxel[..., 0].bool()
    
    def get_contact_mask(self) -> torch.Tensor:
        """Get contact mask (40, 40, 40)."""
        if self.voxel is None:
            raise ValueError("Voxel data not initialized")
        return self.voxel[..., 1].bool()
    
    def get_thumb_contact_mask(self) -> torch.Tensor:
        """Get thumb contact mask (40, 40, 40)."""
        if self.voxel is None:
            raise ValueError("Voxel data not initialized")
        return self.voxel[..., 2].bool()
    
    def get_contact_points(self, use_cache: bool = True) -> torch.Tensor:
        """
        Get contact points in hand coordinate.
        
        Args:
            use_cache: Whether to use cached result
        
        Returns:
            (N, 3) contact point cloud
        """
        if use_cache and self._contact_points_cache is not None:
            return self._contact_points_cache
        
        contact_mask = self.get_contact_mask()
        indices = torch.nonzero(contact_mask, as_tuple=False)
        
        bound = self.config.bound
        resolution = self.config.resolution
        
        points = indices.float() * resolution + torch.tensor(
            [-bound + resolution/2, -bound + resolution/2, -bound + resolution/2],
            device=self.device
        )
        
        if use_cache:
            self._contact_points_cache = points
        
        return points
    
    def get_thumb_contact_points(self, use_cache: bool = True) -> torch.Tensor:
        """
        Get thumb contact points in hand coordinate.
        
        Args:
            use_cache: Whether to use cached result
        
        Returns:
            (N, 3) thumb contact point cloud
        """
        if use_cache and self._thumb_contact_points_cache is not None:
            return self._thumb_contact_points_cache
        
        thumb_mask = self.get_thumb_contact_mask()
        indices = torch.nonzero(thumb_mask, as_tuple=False)
        
        bound = self.config.bound
        resolution = self.config.resolution
        
        points = indices.float() * resolution + torch.tensor(
            [-bound + resolution/2, -bound + resolution/2, -bound + resolution/2],
            device=self.device
        )
        
        if use_cache:
            self._thumb_contact_points_cache = points
        
        return points
    
    def get_non_contact_ibs_points(self) -> torch.Tensor:
        """Get IBS points that are not contact points."""
        occupancy = self.get_occupancy_mask()
        contact = self.get_contact_mask()
        non_contact_mask = occupancy & ~contact
        
        indices = torch.nonzero(non_contact_mask, as_tuple=False)
        
        bound = self.config.bound
        resolution = self.config.resolution
        
        points = indices.float() * resolution + torch.tensor(
            [-bound + resolution/2, -bound + resolution/2, -bound + resolution/2],
            device=self.device
        )
        
        return points
    
    # ==================== Statistics ====================
    
    def num_ibs_points(self) -> int:
        """Get number of IBS points."""
        return int(self.get_occupancy_mask().sum().item())
    
    def num_contact_points(self) -> int:
        """Get number of contact points."""
        return int(self.get_contact_mask().sum().item())
    
    def num_thumb_contact_points(self) -> int:
        """Get number of thumb contact points."""
        return int(self.get_thumb_contact_mask().sum().item())
    
    def get_statistics(self) -> dict:
        """Get statistics about this IBS."""
        return {
            'num_ibs_points': self.num_ibs_points(),
            'num_contact_points': self.num_contact_points(),
            'num_thumb_contact_points': self.num_thumb_contact_points(),
            'voxel_shape': tuple(self.voxel.shape) if self.voxel is not None else None,
            'has_w2h_trans': self.w2h_trans is not None,
            'has_hand_dis': self.hand_dis is not None,
        }
    
    # ==================== Manipulation ====================
    
    def to(self, device: str) -> 'IBS':
        """Move IBS to device."""
        self.device = device
        if self.voxel is not None:
            self.voxel = self.voxel.to(device)
        if self.w2h_trans is not None:
            self.w2h_trans = self.w2h_trans.to(device)
        if self.hand_dis is not None:
            self.hand_dis = self.hand_dis.to(device)
        # Clear cache
        self._point_cloud_cache = None
        self._contact_points_cache = None
        self._thumb_contact_points_cache = None
        return self
    
    def clone(self) -> 'IBS':
        """Create a deep copy of this IBS."""
        return IBS(
            voxel=self.voxel.clone() if self.voxel is not None else None,
            w2h_trans=self.w2h_trans.clone() if self.w2h_trans is not None else None,
            hand_dis=self.hand_dis.clone() if self.hand_dis is not None else None,
            config=self.config,
            device=self.device
        )
    
    def clear_cache(self):
        """Clear all cached data."""
        self._point_cloud_cache = None
        self._contact_points_cache = None
        self._thumb_contact_points_cache = None


class IBSBatch:
    """
    Batch of IBS objects for efficient processing.
    
    Attributes:
        voxels: (B, 40, 40, 40, 3) batched voxel data
        w2h_trans: (B, 4, 4) batched transformations
        config: IBSConfig
        device: torch device
    """
    
    def __init__(
        self,
        voxels: Optional[Union[np.ndarray, torch.Tensor]] = None,
        w2h_trans: Optional[Union[np.ndarray, torch.Tensor]] = None,
        hand_dis: Optional[Union[np.ndarray, torch.Tensor]] = None,
        config: Optional[IBSConfig] = None,
        device: str = 'cpu'
    ):
        """
        Initialize IBSBatch.
        
        Args:
            voxels: (B, 40, 40, 40, 3) batched voxel data
            w2h_trans: (B, 4, 4) batched transformations
            hand_dis: (B, 40, 40, 40) batched hand distances
            config: IBSConfig
            device: torch device
        """
        self.config = config or IBSConfig()
        self.device = device
        
        if voxels is not None:
            if isinstance(voxels, np.ndarray):
                voxels = torch.from_numpy(voxels)
            self.voxels = voxels.to(device)
            self.batch_size = voxels.shape[0]
        else:
            self.voxels = None
            self.batch_size = 0
        
        if w2h_trans is not None:
            if isinstance(w2h_trans, np.ndarray):
                w2h_trans = torch.from_numpy(w2h_trans)
            self.w2h_trans = w2h_trans.float().to(device)
        else:
            self.w2h_trans = None
        
        if hand_dis is not None:
            if isinstance(hand_dis, np.ndarray):
                hand_dis = torch.from_numpy(hand_dis)
            self.hand_dis = hand_dis.to(device)
        else:
            self.hand_dis = None
    
    @classmethod
    def from_file(
        cls,
        ibs_path: str,
        w2h_trans_path: str,
        hand_dis_path: Optional[str] = None,
        device: str = 'cpu'
    ) -> 'IBSBatch':
        """Load IBSBatch from numpy files."""
        voxels = np.load(ibs_path)
        w2h_trans = np.load(w2h_trans_path)
        
        hand_dis = None
        if hand_dis_path is not None and os.path.exists(hand_dis_path):
            hand_dis = np.load(hand_dis_path)
        
        return cls(voxels=voxels, w2h_trans=w2h_trans, hand_dis=hand_dis, device=device)
    
    @classmethod
    def from_ibs_list(cls, ibs_list: List[IBS], device: str = 'cpu') -> 'IBSBatch':
        """Create IBSBatch from list of IBS objects."""
        if len(ibs_list) == 0:
            return cls(device=device)
        
        voxels = torch.stack([ibs.voxel for ibs in ibs_list], dim=0)
        w2h_trans = torch.stack([ibs.w2h_trans for ibs in ibs_list], dim=0)
        
        hand_dis = None
        if all(ibs.hand_dis is not None for ibs in ibs_list):
            hand_dis = torch.stack([ibs.hand_dis for ibs in ibs_list], dim=0)
        
        config = ibs_list[0].config
        
        return cls(voxels=voxels, w2h_trans=w2h_trans, hand_dis=hand_dis, config=config, device=device)
    
    def __len__(self) -> int:
        return self.batch_size
    
    def __getitem__(self, idx: int) -> IBS:
        """Get single IBS from batch."""
        return IBS(
            voxel=self.voxels[idx] if self.voxels is not None else None,
            w2h_trans=self.w2h_trans[idx] if self.w2h_trans is not None else None,
            hand_dis=self.hand_dis[idx] if self.hand_dis is not None else None,
            config=self.config,
            device=self.device
        )
    
    def to_network_input(self) -> torch.Tensor:
        """
        Convert to network input format (B, C, D, H, W).
        
        Returns:
            (B, 2, 40, 40, 40) tensor
        """
        if self.voxels is None:
            raise ValueError("Voxel data not initialized")
        
        # Get channels
        occupancy = self.voxels[..., 0].float()
        contact = self.voxels[..., 1].float()
        thumb_contact = self.voxels[..., 2].float()
        
        # Network format
        ch0 = torch.where(occupancy > 0, 1.0, -1.0)
        ch0 = torch.where(contact > 0, 2.0, ch0)
        ch1 = thumb_contact * 2.0
        
        return torch.stack([ch0, ch1], dim=1)  # (B, 2, 40, 40, 40)
    
    def save(self, ibs_path: str, w2h_trans_path: str, hand_dis_path: Optional[str] = None):
        """Save IBSBatch to files."""
        if self.voxels is not None:
            np.save(ibs_path, self.voxels.cpu().numpy())
        if self.w2h_trans is not None:
            np.save(w2h_trans_path, self.w2h_trans.cpu().numpy())
        if hand_dis_path is not None and self.hand_dis is not None:
            np.save(hand_dis_path, self.hand_dis.cpu().numpy())
    
    def to(self, device: str) -> 'IBSBatch':
        """Move to device."""
        self.device = device
        if self.voxels is not None:
            self.voxels = self.voxels.to(device)
        if self.w2h_trans is not None:
            self.w2h_trans = self.w2h_trans.to(device)
        if self.hand_dis is not None:
            self.hand_dis = self.hand_dis.to(device)
        return self
    
    def get_statistics(self) -> dict:
        """Get batch statistics."""
        if self.voxels is None:
            return {'batch_size': 0}
        
        occupancy = self.voxels[..., 0].bool()
        contact = self.voxels[..., 1].bool()
        thumb = self.voxels[..., 2].bool()
        
        return {
            'batch_size': self.batch_size,
            'total_ibs_points': int(occupancy.sum().item()),
            'avg_ibs_points': float(occupancy.sum().item()) / self.batch_size,
            'total_contact_points': int(contact.sum().item()),
            'avg_contact_points': float(contact.sum().item()) / self.batch_size,
            'total_thumb_points': int(thumb.sum().item()),
            'avg_thumb_points': float(thumb.sum().item()) / self.batch_size,
        }
