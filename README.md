# HumanBlender
Rendering 3D models using Blender

## Installation

Create and activate a virtual environment
```console
python -m venv env
source env/bin/activate
```

Install the required libraries: BLENDER Python API [bpy](https://docs.blender.org/api/current/index.html), [Open3D](http://www.open3d.org/) and [Numpy]([https://matplotlib.org/](https://numpy.org/))
```console
pip install bpy
pip install open3d
pip install numpy
```

## Usage
The input data to render must be in .ply format.
>REMARK: input_data/ and output_data/ provide some sample input and output files

To render multiple views of the 3D shape:
```console
python render_pcd.py --pc_path <path_to_ply> --views <n_views> --out_path <path_to_output_folder>
```

To get an AVI animation of the rotating 3D shape:
```console
python render_pcd.py --pc_path <path_to_ply> --animation  --out_path <path_to_output_folder>
```

**Additional arguments**: 
* _frames_: number of frames in the animation
* _pt_size_: size of the points
* _color_: color of the points (if the point cloud is not colored)
