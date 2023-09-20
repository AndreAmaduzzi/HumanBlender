import math
from pathlib import Path
from typing import List

import bpy
import numpy as np
import open3d as o3d
import argparse
import os

from utils import (
    add_track_to_constraint,
    create_camera,
    create_light,
    create_material,
    create_plane,
    pcd_to_sphere,
    remove_objects,
    set_camera_params,
    set_engine_params,
    set_principled_node,
    set_render_params,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pc_path', type=str, default='', required=True, help='path of the PLY point cloud to render')
    parser.add_argument('--animation', action="store_true", help='if set, the output is an AVI video with the rotating cloud')
    parser.add_argument('--views', type=int, default=10, help="number of views to render, if not animation")
    parser.add_argument('--pt_size', type=float, default=0.01, help='size of each point of the cloud')
    parser.add_argument('--color', type=tuple, default=(1.0, 0.0, 0.0, 1.0), help='color of the points')
    parser.add_argument('--out_path', type=str, default='', required=True, help='output folder')


    args = parser.parse_args()
    return args

def main():
    
    args = parse_args()
    animation = args.animation
    path_input = Path(args.pc_path)
    base_color = args.color     # color of the points
    pt_size = args.pt_size      # size of the points      
    

    path_out = Path(args.out_path)
    path_out.mkdir(exist_ok=True, parents=True)

    # Read from hesiod
    num_samples = 100
    res_x = int(800)
    res_y = int(800)
    devices = [0]
    location_camera = (0, 4.0, 1.0)
    loc_light = (0, 0, 2)
    rot_light = (math.radians(0), math.radians(0), math.radians(0))
    energy = 3.0
    rot_object = (math.radians(0), math.radians(0), math.radians(0))    # Adjust here the rotation of the cloud, if not correct in the output
    add_plane = False
    devices = [0]
    save_blender = True
    use_denoiser = True
    lens = 85
    plane_only_shadow = False

    # Reset
    remove_objects()

    # Object
    pcd = o3d.io.read_point_cloud(str(path_input))
    pts = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    if len(colors):
        pts = np.concatenate((pts, colors), axis=1)

    focus_target_object = pcd_to_sphere(pts, radius=pt_size, scale=1)  # radius: size of each point    # type: ignore 

    if pts.shape[1] > 3:
        mat = create_material("Material_Visualization", use_nodes=True, make_node_tree_empty=True)
        output_node = mat.node_tree.nodes.new(type="ShaderNodeOutputMaterial")
        principled_node = mat.node_tree.nodes.new(type="ShaderNodeBsdfPrincipled")
        rgb_node = mat.node_tree.nodes.new(type="ShaderNodeRGB")
        mix_node = mat.node_tree.nodes.new(type="ShaderNodeMixShader")
        attrib_node = mat.node_tree.nodes.new(type="ShaderNodeAttribute")
        attrib_node.attribute_name = "Col"
        rgb_node.outputs["Color"].default_value = (0.1, 0.1, 0.1, 1.0)

        mat.node_tree.links.new(attrib_node.outputs["Color"], principled_node.inputs["Base Color"])
        mat.node_tree.links.new(principled_node.outputs["BSDF"], mix_node.inputs[1])
        mat.node_tree.links.new(mix_node.outputs["Shader"], output_node.inputs["Surface"])
    else:
        # Material
        mat = create_material("Material_Right", use_nodes=True, make_node_tree_empty=True)
        output_node = mat.node_tree.nodes.new(type="ShaderNodeOutputMaterial")
        principled_node = mat.node_tree.nodes.new(type="ShaderNodeBsdfPrincipled")
        set_principled_node(principled_node, base_color=base_color)

        mat.node_tree.links.new(principled_node.outputs["BSDF"], output_node.inputs["Surface"])

    focus_target_object.data.materials.append(mat)
    # Location Plane
    if add_plane:
        z_plane = (focus_target_object.dimensions[-1] * 0.5) + 0.1
        loc_plane = (0.0, 0.0, -z_plane)
        create_plane(size=100.0, location=loc_plane)
        bpy.context.object.cycles.is_shadow_catcher = plane_only_shadow

    # Camera
    camera_object = create_camera(location=location_camera)
    add_track_to_constraint(camera_object, focus_target_object)
    set_camera_params(camera_object.data, focus_target_object, lens=lens)
    scene = bpy.data.scenes["Scene"]
    scene.camera = camera_object

    # Light
    light = create_light(location=loc_light, rotation=rot_light, name="sun", energy=energy)
    bpy.context.collection.objects.link(light)

    # Render Setting
    path_render = path_out / f"{path_input.stem}.png"
    set_render_params(
        scene, path_render, resolution_x=res_x, resolution_y=res_y, use_transparent_bg=True
    )
    set_engine_params(
        scene, ids_cuda_devices=devices, num_samples=num_samples, use_denoiser=use_denoiser
    )

    obj = bpy.data.objects["object"]
    
    if animation:
        obj.rotation_mode = 'XYZ'
        scene.frame_start = 1
        scene.frame_end = 400
        obj.rotation_euler = rot_object
        obj.keyframe_insert('rotation_euler', index=-1 ,frame=scene.frame_start)
        obj.rotation_euler = (0, 0, math.radians(360))
        obj.keyframe_insert('rotation_euler', index=-1 ,frame=scene.frame_end)

        scene.render.filepath = f"{path_out}/{path_input.stem}"
        scene.render.image_settings.file_format = "AVI_JPEG"
        scene.render.film_transparent = True
    else:
        stepsize = 360.0 / args.views
        for i in range(0, args.views):
            print("Rotation {}, {}".format((stepsize * i), math.radians(stepsize * i)))

            filename = os.path.splitext(os.path.basename(args.pc_path))[0]
            render_file_path = f"{path_out}/{filename}_{i}"

            scene.render.filepath = render_file_path
            print('render file path: ', render_file_path)
            
            # Uncomment to get depth, normal, albedo, id
            #depth_file_output.file_slots[0].path = render_file_path + "_depth"
            #normal_file_output.file_slots[0].path = render_file_path + "_normal"
            #albedo_file_output.file_slots[0].path = render_file_path + "_albedo"
            #id_file_output.file_slots[0].path = render_file_path + "_id"

            print('rendering...')
            bpy.ops.render.render(write_still=True)  # render still
            print('save')
            bpy.ops.wm.save_mainfile()

            obj.rotation_euler[2] += math.radians(stepsize)
        #obj.rotation_euler = rot_object

    bpy.ops.render.render(write_still=True, animation=animation)

    if save_blender:
        bpy.ops.wm.save_mainfile()
    bpy.ops.wm.read_factory_settings()


if __name__ == "__main__":
    main()
