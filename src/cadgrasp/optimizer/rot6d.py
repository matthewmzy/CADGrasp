import torch
import numpy as np
import transforms3d


def get_rot6d_from_rot3d(rot3d):
    global_rotation = np.array(transforms3d.euler.euler2mat(rot3d[0], rot3d[1], rot3d[2]))
    return global_rotation.T.reshape(9)[:6]


def compute_rotation_matrix_from_ortho6d(poses):
    """
    Code from
    https://github.com/papagina/RotationContinuity
    On the Continuity of Rotation Representations in Neural Networks
    Zhou et al. CVPR19
    https://zhouyisjtu.github.io/project_rotation/rotation.html
    """
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3
        
    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3
        
    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix


def robust_compute_rotation_matrix_from_ortho6d(poses):
    """
    Instead of making 2nd vector orthogonal to first
    create a base that takes into account the two predicted
    directions equally
    """
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    y = normalize_vector(y_raw)  # batch*3
    middle = normalize_vector(x + y)
    orthmid = normalize_vector(x - y)
    x = normalize_vector(middle + orthmid)
    y = normalize_vector(middle - orthmid)
    # Their scalar product should be small !
    # assert torch.einsum("ij,ij->i", [x, y]).abs().max() < 0.00001
    z = normalize_vector(cross_product(x, y))

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    # Check for reflection in matrix ! If found, flip last vector TODO
    # assert (torch.stack([torch.det(mat) for mat in matrix ])< 0).sum() == 0
    return matrix

def robust_compute_rotation_matrix_from_ortho6d_np(poses):
    # Split into two 3D vectors
    x_raw = poses[:, 0:3]  # (N, 3)
    y_raw = poses[:, 3:6]  # (N, 3)
    
    # Step 1: Normalize both input vectors
    x = x_raw / np.maximum(np.linalg.norm(x_raw, axis=1, keepdims=True), 1e-8)
    y = y_raw / np.maximum(np.linalg.norm(y_raw, axis=1, keepdims=True), 1e-8)
    
    # Step 2: Create balanced middle vectors
    middle = (x + y) / np.maximum(np.linalg.norm(x + y, axis=1, keepdims=True), 1e-8)
    orthmid = (x - y) / np.maximum(np.linalg.norm(x - y, axis=1, keepdims=True), 1e-8)
    
    # Step 3: Construct orthogonal basis
    x_new = (middle + orthmid) / np.maximum(np.linalg.norm(middle + orthmid, axis=1, keepdims=True), 1e-8)
    y_new = (middle - orthmid) / np.maximum(np.linalg.norm(middle - orthmid, axis=1, keepdims=True), 1e-8)
    
    # Step 4: Compute final z-axis
    z = np.cross(x_new, y_new, axis=1)
    z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-8)
    
    # Step 5: Construct rotation matrix
    return np.stack([x_new, y_new, z], axis=2)


def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(v_mag, v.new([1e-8]))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v/v_mag
    return v


def cross_product(u, v):
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        
    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)
        
    return out


def add_transform_perturbation(transforms, max_translation=0.015, max_rotation=5):
    """
    Add random translation and rotation perturbations to transformation matrices.
    
    Args:
        transforms (np.ndarray or torch.Tensor): Transformation matrices of shape (N, 4, 4).
        max_translation (float): Maximum translation perturbation value (unit: m).
        max_rotation (float): Maximum rotation perturbation value (unit: degrees).
    
    Returns:
        np.ndarray or torch.Tensor: Perturbed transformation matrices, same type as input.
    """
    # Determine if input is NumPy or PyTorch and set backend accordingly
    is_torch = isinstance(transforms, torch.Tensor)
    N = transforms.shape[0]
    
    # Use appropriate random number generation and operations based on input type
    if is_torch:
        device = transforms.device
        randn = lambda *args: torch.randn(*args, device=device)
        clip = torch.clamp
        eye = torch.eye
        matmul = torch.matmul
        zeros = torch.zeros
        radians = torch.deg2rad
        cos = torch.cos
        sin = torch.sin
    else:
        randn = np.random.normal
        clip = np.clip
        eye = np.eye
        matmul = np.matmul
        zeros = np.zeros
        radians = np.radians
        cos = np.cos
        sin = np.sin

    # 1. Generate translation perturbation
    translation_perturbation = randn(N, 3) * (max_translation / 3)  # Normal distribution, std = max_translation / 3
    translation_perturbation = clip(translation_perturbation, -max_translation, max_translation)
    
    # Convert translation perturbation to homogeneous transformation matrices
    translation_matrices = eye(4).repeat(N, 1, 1) if is_torch else eye(4)[np.newaxis, :, :].repeat(N, axis=0)
    translation_matrices[:, :3, 3] = translation_perturbation
    
    # 2. Generate rotation perturbation
    rotation_perturbation = randn(N, 3) * (max_rotation / 3)  # Normal distribution, std = max_rotation / 3
    rotation_perturbation = clip(rotation_perturbation, -max_rotation, max_rotation)
    
    # Convert rotation perturbation to rotation matrices
    rotation_matrices = zeros((N, 4, 4)) if is_torch else zeros((N, 4, 4))
    rotation_perturbation_rad = radians(rotation_perturbation)  # Convert degrees to radians
    
    for i in range(N):
        rx, ry, rz = rotation_perturbation_rad[i]
        Rx = zeros((3, 3)) if is_torch else np.zeros((3, 3))
        Rx[0, 0] = 1
        Rx[1, 1] = cos(rx)
        Rx[1, 2] = -sin(rx)
        Rx[2, 1] = sin(rx)
        Rx[2, 2] = cos(rx)
        
        Ry = zeros((3, 3)) if is_torch else np.zeros((3, 3))
        Ry[0, 0] = cos(ry)
        Ry[0, 2] = sin(ry)
        Ry[1, 1] = 1
        Ry[2, 0] = -sin(ry)
        Ry[2, 2] = cos(ry)
        
        Rz = zeros((3, 3)) if is_torch else np.zeros((3, 3))
        Rz[0, 0] = cos(rz)
        Rz[0, 1] = -sin(rz)
        Rz[1, 0] = sin(rz)
        Rz[1, 1] = cos(rz)
        Rz[2, 2] = 1
        
        R = matmul(matmul(Rz, Ry), Rx)  # Combine rotation matrices
        rotation_matrices[i, :3, :3] = R
        rotation_matrices[i, 3, 3] = 1
    
    # 3. Combine translation and rotation perturbations
    perturbation_matrices = matmul(translation_matrices, rotation_matrices)
    if is_torch:
        perturbation_matrices = perturbation_matrices.to(device)
    # 4. Apply perturbation to original transformation matrices
    perturbed_transforms = matmul(perturbation_matrices, transforms)
    return perturbed_transforms