# LEAP Hand Model Editing Tools

Interactive GUI tools for visualizing and editing LEAP Hand model parameters, including self-collision keypoints and palmar surface points.

## Tools Overview

| Script | Description |
|--------|-------------|
| `visualize_keypoints_gui.py` | Full GUI editor for penetration keypoints |
| `palmar_surface_collector.py` | Lasso selection tool for palmar surface points |
| `visualize_keypoints_plotly.py` | Browser-based 3D visualization |
| `visualize_keypoints_simple.py` | Lightweight Open3D visualizer |

---

## 1. visualize_keypoints_gui.py - Penetration Keypoints Editor

Full-featured GUI for editing self-collision detection keypoints.

**Features:**
- Left panel: Link selection, keypoint management, offset adjustment
- Real-time joint angle adjustment via sliders
- Real-time keypoint X/Y/Z offset adjustment
- Red sphere = selected keypoint, Blue = unselected
- Transparent spheres show collision radius (0.025m)

**Usage:**
```bash
python tests/hand_info/visualize_keypoints_gui.py
```

**GUI Controls:**
- Select link from dropdown menu
- Add/delete keypoints with buttons
- Adjust keypoint position with X/Y/Z sliders
- Save changes to JSON file

---

## 2. palmar_surface_collector.py - Palmar Surface Editor

Lasso selection tool for defining palmar (inner) surface points on the hand mesh.

**Features:**
- Draw lasso to select multiple mesh vertices
- Supports both ADD and REMOVE selection modes
- Real-time visualization of selected points
- Save selected vertices as palmar surface points

**Usage:**
```bash
python tests/hand_info/palmar_surface_collector.py
```

**Controls:**
- **Left-click + drag**: Draw lasso selection
- **Mode toggle**: Switch between ADD/REMOVE modes
- **Save button**: Export selected vertices

**Output:**
- Saves to `robot_models/meta/leap_hand/palmar_surface_vertices.json`

---

## 3. visualize_keypoints_plotly.py - Browser Visualization

Interactive 3D visualization in web browser using Plotly.

**Features:**
- View hand model with all keypoints
- Each link's keypoints shown in different colors
- Hover to see keypoint details
- Export to HTML file

**Usage:**
```bash
# Basic visualization
python tests/hand_info/visualize_keypoints_plotly.py

# With custom joint angles (radians, comma-separated)
python tests/hand_info/visualize_keypoints_plotly.py --joint_angles "0.5,0.3,0.2,..."

# Export to HTML
python tests/hand_info/visualize_keypoints_plotly.py --output output.html
```

---

## 4. visualize_keypoints_simple.py - Lightweight Visualizer

Simple Open3D visualizer with keyboard shortcuts.

**Usage:**
```bash
python tests/hand_info/visualize_keypoints_simple.py
```

**Keyboard Shortcuts:**
| Key | Action |
|-----|--------|
| H | Show help |
| S | Save keypoints |
| Q | Quit |
| R | Reset joint angles |
| J | Modify joint angles |
| A | Add keypoint |
| D | Delete keypoint |
| E | Edit keypoint offset |
| L/K | Switch link |
| N/P | Switch keypoint |

---

## Data File Formats

### Penetration Keypoints

Location: `robot_models/meta/leap_hand/penetration_keypoints.json`

```json
{
    "link_name": [
        [x1, y1, z1],
        [x2, y2, z2],
        ...
    ],
    "another_link": [...]
}
```

- Coordinates are in link-local frame (meters)
- Collision radius: 0.025m (used in `HandModel.get_self_penetration()`)

### Palmar Surface Vertices

Location: `robot_models/meta/leap_hand/palmar_surface_vertices.json`

```json
{
    "link_name": [vertex_idx1, vertex_idx2, ...],
    "another_link": [...]
}
```

- Vertex indices reference the link's mesh vertices
- Used for contact point filtering in optimization

---

## LEAP Hand Links

| Link Name | Description |
|-----------|-------------|
| hand_base_link | Palm base |
| mcp_joint | Index MCP joint |
| pip | Index PIP joint |
| dip | Index DIP joint |
| fingertip | Index fingertip |
| mcp_joint_2 | Middle MCP joint |
| pip_2 | Middle PIP joint |
| dip_2 | Middle DIP joint |
| fingertip_2 | Middle fingertip |
| mcp_joint_3 | Ring MCP joint |
| pip_3 | Ring PIP joint |
| dip_3 | Ring DIP joint |
| fingertip_3 | Ring fingertip |
| pip_4 | Thumb base |
| thumb_pip | Thumb PIP |
| thumb_dip | Thumb DIP |
| thumb_fingertip | Thumb fingertip |

---

## Dependencies

```bash
pip install open3d numpy trimesh pytorch-kinematics transforms3d plotly
```

---

## Notes

1. **Collision Radius**: Fixed at 0.025m, matching `HandModel.get_self_penetration()` threshold

2. **Coordinate System**: All offsets are in link-local coordinates, not world coordinates

3. **Remember to Save**: Press S or click Save button after making changes

4. **File Paths**: 
   - URDF: `robot_models/urdf/leap_hand.urdf`
   - Keypoints: `robot_models/meta/leap_hand/penetration_keypoints.json`
   - Palmar surface: `robot_models/meta/leap_hand/palmar_surface_vertices.json`
