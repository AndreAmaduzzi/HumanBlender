# ==============================================================================
# Copyright (c) 2023 Tiange Luo, tiange.cs@gmail.com
# Last modified: June 22, 2023
#
# This code is licensed under the MIT License.
# ==============================================================================

### from https://github.com/crockwell/Cap3D/blob/main/captioning_pipeline/render_script.py

import bpy
import sys
import mathutils
from mathutils import Vector, Matrix, Euler
import argparse
import numpy as np
import math
import os
import time
import pickle
from PIL import Image
import random

## solve the division problem
from decimal import Decimal, getcontext
getcontext().prec = 28  # Set the precision for the decimal calculations.

parser = argparse.ArgumentParser()
parser.add_argument('--object_path_pkl', type = str, required = True)
parser.add_argument("--parent_dir", type = str, default='./example_material')

argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

uid_paths = pickle.load(open(args.object_path_pkl, 'rb'))
# random.shuffle(uids)


bpy.context.scene.render.engine = 'CYCLES'
# small samples for fast rendering
bpy.context.scene.cycles.samples = 16
# bpy.context.scene.cycles.samples = 128
bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
bpy.context.scene.cycles.device = 'GPU'
bpy.context.scene.cycles.use_denoising = True
bpy.context.scene.cycles.denoiser = 'OPTIX'
for scene in bpy.data.scenes:
    scene.cycles.device = 'GPU'

# get_devices() to let Blender detects GPU device
bpy.context.preferences.addons["cycles"].preferences.get_devices()
print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
for d in bpy.context.preferences.addons["cycles"].preferences.devices:
    if 'NVIDIA' in d['name']:
        d["use"] = 1 # Using all devices, include GPU and CPU
        print(d["name"], d["use"])
    else:
        d["use"] = 0 # Using all devices, include GPU and CPU
        print(d["name"], d["use"])

render_prefs = bpy.context.preferences.addons['cycles'].preferences
render_device_type = render_prefs.compute_device_type
compute_device_type = render_prefs.devices[0].type if len(render_prefs.devices) > 0 else None
# Check if the compute device type is GPU
if render_device_type == 'CUDA' and compute_device_type == 'CUDA':
    # GPU is being used for rendering
    print("Using GPU for rendering")
else:
    # GPU is not being used for rendering
    print("Not using GPU for rendering")


# if the object is too far away from the origin, pull it closer
def check_object_location(mesh_objects, max_distance):
        return True

# compute the bounding box of the mesh objects
def compute_bounding_box(mesh_objects):
    return bbox_center, bbox_size

# normalize objects 
def normalize_and_center_objects(mesh_objects, normalization_range):
    return bbox_center, bbox_size

# check if rendered object will cross the boundary of the image
def project_points_to_camera_space(obj, camera):
    return bbox_image

# prepare the scene
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.data.objects['Cube'].select_set(True)
bpy.ops.object.delete()

# Create lights

bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='LIGHT')
bpy.ops.object.delete()

def create_light(name, light_type, energy, location, rotation):
    bpy.ops.object.light_add(type=light_type, align='WORLD', location=location, scale=(1, 1, 1))
    light = bpy.context.active_object
    light.name = name
    light.data.energy = energy
    light.rotation_euler = rotation
    return light

def three_point_lighting():
    
    # Key ligh
    key_light = create_light(
        name="KeyLight",
        light_type='AREA',
        energy=1000,
        location=(4, -4, 4),
        rotation=(math.radians(45), 0, math.radians(45))
    )
    key_light.data.size = 2

    # Fill light
    fill_light = create_light(
        name="FillLight",
        light_type='AREA',
        energy=300,
        location=(-4, -4, 2),
        rotation=(math.radians(45), 0, math.radians(135))
    )
    fill_light.data.size = 2

    # Rim/Back light
    rim_light = create_light(
        name="RimLight",
        light_type='AREA',
        energy=600,
        location=(0, 4, 0),
        rotation=(math.radians(45), 0, math.radians(225))
    )
    rim_light.data.size = 2

three_point_lighting()

for i in range(8):
    os.makedirs(os.path.join(args.parent_dir, 'Cap3D_imgs', 'Cap3D_imgs_view%d'%i), exist_ok=True)
    os.makedirs(os.path.join(args.parent_dir, 'Cap3D_imgs', 'Cap3D_imgs_view%d_CamMatrix'%i), exist_ok=True)
os.makedirs(os.path.join(args.parent_dir, 'Cap3D_captions'), exist_ok=True)

def load_ply(filepath):
    import plyfile
    plydata = plyfile.PlyData.read(filepath)
    
    verts = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
    faces = np.vstack(plydata['face']['vertex_index'])
    vertex_colors = np.vstack([plydata['vertex']['red'], plydata['vertex']['green'], plydata['vertex']['blue']]).T / 255

    mesh = bpy.data.meshes.new(name="Imported PLY")
    mesh.from_pydata(verts.tolist(), [], faces.tolist())

    # create color layer
    color_layer = mesh.vertex_colors.new()

    # assign colors to vertices
    for poly in mesh.polygons:
        for loop_index in poly.loop_indices:
            loop_vert_index = mesh.loops[loop_index].vertex_index
            color_layer.data[loop_index].color = vertex_colors[loop_vert_index].tolist() + [1.0]

    # create new material
    mat = bpy.data.materials.new(name="VertexCol")
    
    # enable 'use_nodes'
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    
    # get the 'Material Output' node
    material_output = nodes.get('Material Output')
    
    # add 'Vertex Color' node
    vertex_color_node = nodes.new(type='ShaderNodeVertexColor')
    
    # add 'BSDF' node
    bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')
    
    # link 'Vertex Color' node to 'BSDF' node
    mat.node_tree.links.new(vertex_color_node.outputs['Color'], bsdf_node.inputs['Base Color'])
    
    # link 'BSDF' node to 'Material Output' node
    mat.node_tree.links.new(bsdf_node.outputs['BSDF'], material_output.inputs['Surface'])

    # Create new object and link mesh and material
    obj = bpy.data.objects.new("ImportedPLY", mesh)
    obj.data.materials.append(mat)

    # Link object to the current collection
    bpy.context.collection.objects.link(obj)

    return mesh

for uid_path in uid_paths:
    if not os.path.exists(uid_path):
        continue

    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()
    
    _, ext = os.path.splitext(uid_path)
    ext = ext.lower()
    if ext in [".glb", ".gltf"]:
        bpy.ops.import_scene.gltf(filepath=uid_path)
    elif ext == '.obj':
        bpy.ops.import_scene.obj(filepath=uid_path)


    print('begin*************')
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']

    for mesh_obj in mesh_objects:
        # Create a new material
        mat = bpy.data.materials.new(name="VertexColMaterial")
        mesh_obj.data.materials.clear()
        mesh_obj.data.materials.append(mat)

        # Use 'Use nodes':
        mat.use_nodes = True
        nodes = mat.node_tree.nodes

        # Clear default nodes
        for node in nodes:
            nodes.remove(node)

        # Add a Vertex Color Node and a Diffuse BSDF shader
        vertex_color_node = nodes.new(type='ShaderNodeVertexColor')
        emission_shader = nodes.new(type='ShaderNodeEmission')
        output_shader = nodes.new(type='ShaderNodeOutputMaterial')

        # Connect the Vertex Color node to the Emission Shader
        mat.node_tree.links.new(vertex_color_node.outputs["Color"], emission_shader.inputs["Color"])
        # Connect the Emission Shader to the Material Output
        mat.node_tree.links.new(emission_shader.outputs["Emission"], output_shader.inputs["Surface"])

        # If your obj specifies a vertex color layer other than 'Col', you can adjust the name here:
        if "Col" in mesh_obj.data.vertex_colors:
            vertex_color_node.layer_name = "Col"

    # Compute the bounding box for the objects
    normalization_range = 1.0
    bbox_center, bbox_size = normalize_and_center_objects(mesh_objects, normalization_range)

    distance = max(bbox_size.x, bbox_size.y, bbox_size.z)
    ratio = 1.15
    elevation_factor = 0.2


    camera = bpy.context.scene.camera
    name = uid_path.split('/')[-1].split('.')[0]
    for camera_opt in range(-1, 8):
        # use transparent background to adjust camera distance
        if camera_opt == -1:
            bpy.context.scene.render.image_settings.color_mode = 'RGBA'
            bpy.context.scene.render.film_transparent = True
            camera.location = Vector((distance * ratio, - distance * ratio, distance * elevation_factor * ratio))
        elif camera_opt == 0:
            img_path = os.path.join(args.parent_dir, 'Cap3D_imgs', 'Cap3D_imgs_view_bg', '%s_bg.png'%(uid_path.split('/')[-1].split('.')[0]))
            img = Image.open(img_path)
            img_array = np.array(img)
            if np.sum(img_array<10) > 1020000:
                print(name, 'WARNING: rendered image may contain too much white space')

            # change to white background to render the final 8 views
            bpy.context.scene.world.node_tree.nodes["Background"].inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
            bpy.context.scene.render.film_transparent = False
            camera.location = Vector((distance * ratio, - distance * ratio, distance * elevation_factor * ratio))

            # check if the object is within the image
            while True:
                flag_list = []
                for obj in mesh_objects:
                    bbox_image = project_points_to_camera_space(obj, camera)
                    if np.max(np.array(bbox_image) > 512) or np.min(np.array(bbox_image) < 0):
                        flag_list.append(0)
                        ratio += 0.1
                        camera.location = Vector((distance * ratio, - distance * ratio, distance * elevation_factor * ratio))
                if len(flag_list) == 0:
                    break
        elif camera_opt == 1:
            camera.location = Vector((- distance * ratio,  distance * ratio, distance * elevation_factor * ratio))
        elif camera_opt == 2:
            elevation_factor = 0.5
            camera.location = Vector((distance * ratio,  -distance * ratio*0.5, distance * elevation_factor * ratio))
        elif camera_opt == 3:
            elevation_factor = 0.7
            camera.location = Vector((- distance * ratio *0.5,  distance * ratio, distance * elevation_factor * ratio))
        elif camera_opt == 4:
            camera.location = Vector((distance * ratio,  distance * ratio, distance * elevation_factor * ratio))
        elif camera_opt == 5:
            camera.location = Vector((-distance * ratio,  -distance * ratio, distance * elevation_factor * ratio))
        elif camera_opt == 6:
            elevation_factor = 0.5
            camera.location = Vector((distance * ratio,  distance * ratio*0.5, -distance * elevation_factor * ratio))
        elif camera_opt == 7:
            elevation_factor = 0.7
            camera.location = Vector((- distance * ratio *0.5,  -distance * ratio, -distance * elevation_factor * ratio))

        # Make the camera point at the bounding box center
        direction = (bbox_center - camera.location).normalized()
        quat = direction.to_track_quat('-Z', 'Y')
        camera.rotation_euler = quat.to_euler()

        camera.data.clip_start = 0.1
        camera.data.clip_end = max(1000, distance * 2)

        print('bbox_center: ', bbox_center)
        print('bbox_size: ', bbox_size)
        print('distance: ', distance)
        print('camera.location: ', camera.location)

        bpy.context.scene.camera = bpy.data.objects['Camera']
        bpy.context.scene.render.resolution_x = 512
        bpy.context.scene.render.resolution_y = 512

        if camera_opt == -1:
            file_path = os.path.join(args.parent_dir, 'Cap3D_imgs', 'Cap3D_imgs_view_bg', '%s_bg.png'%(uid_path.split('/')[-1].split('.')[0]))
            bpy.context.scene.render.filepath = file_path
            if os.path.exists(file_path):
               continue
        else:
            file_path = os.path.join(args.parent_dir, 'Cap3D_imgs', 'Cap3D_imgs_view%d'%camera_opt, '%s_%d.png'%(uid_path.split('/')[-1].split('.')[0], camera_opt))
            bpy.context.scene.render.filepath = file_path
            if os.path.exists(file_path):
               continue

        bpy.ops.render.render(write_still=True)

        def get_3x4_RT_matrix_from_blender(cam):
            # Use matrix_world instead to account for all constraints
            location, rotation = cam.matrix_world.decompose()[0:2]
            R_world2bcam = rotation.to_matrix().transposed()

            # Use location from matrix_world to account for constraints:     
            T_world2bcam = -1*R_world2bcam @ location

            # put into 3x4 matrix
            RT = Matrix((
                R_world2bcam[0][:] + (T_world2bcam[0],),
                R_world2bcam[1][:] + (T_world2bcam[1],),
                R_world2bcam[2][:] + (T_world2bcam[2],)
                ))
            return RT

        if camera_opt>=0:
            RT = get_3x4_RT_matrix_from_blender(camera)
            
            RT_path = os.path.join(args.parent_dir, 'Cap3D_imgs', 'Cap3D_imgs_view%d_CamMatrix'%camera_opt, '%s_%d.npy'%(uid_path.split('/')[-1].split('.')[0], camera_opt))
            if os.path.exists(RT_path):
                continue
            np.save(RT_path, RT)

bpy.ops.wm.quit_blender()