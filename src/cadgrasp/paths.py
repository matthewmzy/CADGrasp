"""
Project path utilities.

This module provides functions to get project-relative paths without using os.chdir().
All paths in the project should be resolved through this module.

Usage:
    from cadgrasp.paths import get_project_root, project_path
    
    # Get project root directory
    root = get_project_root()
    
    # Get absolute path for a project-relative path
    urdf_path = project_path('robot_models/urdf/leap_hand.urdf')
    data_path = project_path('data/DexGraspNet2.0/scenes')
"""

import os
from pathlib import Path
from functools import lru_cache


@lru_cache(maxsize=1)
def get_project_root() -> Path:
    """
    Get the project root directory.
    
    The project root is determined by finding the directory containing pyproject.toml,
    starting from this file's location and going up.
    
    Returns:
        Path to the project root directory
    """
    current = Path(__file__).resolve().parent
    
    # Go up until we find pyproject.toml
    for parent in [current] + list(current.parents):
        if (parent / 'pyproject.toml').exists():
            return parent
    
    # Fallback: assume src/cadgrasp/paths.py structure
    return Path(__file__).resolve().parent.parent.parent


def project_path(*paths: str) -> str:
    """
    Get absolute path for a project-relative path.
    
    Args:
        *paths: Path components relative to project root
        
    Returns:
        Absolute path as string
        
    Example:
        >>> project_path('robot_models/urdf/leap_hand.urdf')
        '/path/to/CADGrasp/robot_models/urdf/leap_hand.urdf'
        >>> project_path('data', 'DexGraspNet2.0', 'scenes')
        '/path/to/CADGrasp/data/DexGraspNet2.0/scenes'
    """
    return str(get_project_root() / Path(*paths))


def ensure_project_path(path: str) -> str:
    """
    Ensure a path is absolute. If relative, resolve it from project root.
    
    Args:
        path: Either absolute or project-relative path
        
    Returns:
        Absolute path as string
    """
    if os.path.isabs(path):
        return path
    return project_path(path)
