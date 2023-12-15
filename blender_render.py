import argparse, sys, os, math, re
from os.path import join
import numpy as np
import bpy
from  mathutils import Matrix, Vector
from glob import glob

def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == 'VERTICAL':
        return sensor_y
    return sensor_x

def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit


def render(data_path, out_path, prev=None):
    format = 'PNG'
    color_depth = '8'
    depth_scale = 1.4
    size = 1024

    # Set up rendering
    context = bpy.context
    scene = bpy.context.scene
    render = bpy.context.scene.render

    render.engine = 'CYCLES'
    render.image_settings.color_mode = 'RGBA' # ('RGB', 'RGBA', ...)
    render.image_settings.color_depth = color_depth # ('8', '16')
    render.image_settings.file_format = format # ('PNG', 'OPEN_EXR', 'JPEG, ...)
    render.resolution_x = size
    render.resolution_y = size
    render.resolution_percentage = 100
    render.film_transparent = True

    world = bpy.data.worlds['World']
    world.use_nodes = True
    bg = world.node_tree.nodes['Background']
    bg.inputs[0].default_value = (1, 1, 1, 1)
    bg.inputs[1].default_value = 1

    scene.use_nodes = True
    scene.view_layers["View Layer"].use_pass_normal = True
    scene.view_layers["View Layer"].use_pass_diffuse_color = True
    scene.view_layers["View Layer"].use_pass_object_index = True

    nodes = bpy.context.scene.node_tree.nodes
    links = bpy.context.scene.node_tree.links

    for n in nodes:
        nodes.remove(n)

    for n in links:
        links.remove(n)

    # Create input render layer node
    render_layers = nodes.new('CompositorNodeRLayers')

    scale_node = nodes.new(type="CompositorNodeMixRGB")
    scale_node.blend_type = 'MULTIPLY'
    # scale_node.use_alpha = True
    scale_node.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
    links.new(render_layers.outputs['Normal'], scale_node.inputs[1])

    bias_node = nodes.new(type="CompositorNodeMixRGB")
    bias_node.blend_type = 'ADD'
    # bias_node.use_alpha = True
    bias_node.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
    links.new(scale_node.outputs[0], bias_node.inputs[1])

    normal_file_output = nodes.new(type="CompositorNodeOutputFile")
    normal_file_output.label = 'Normal Output'
    normal_file_output.base_path = ''
    normal_file_output.file_slots[0].use_node_format = True
    normal_file_output.format.file_format = format
    links.new(bias_node.outputs[0], normal_file_output.inputs[0])

    # Delete default cube
    if prev is None:
        context.active_object.select_set(True)
        bpy.ops.object.delete()
        add_light = True
    else:
        prev.select_set(True)
        bpy.ops.object.delete()
        add_light = False

    # Import textured mesh
    bpy.ops.object.select_all(action='DESELECT')

    obj = join(data_path, 'tmp.obj')
    bpy.ops.import_scene.obj(filepath=obj)

    obj = bpy.context.selected_objects[0]
    context.view_layer.objects.active = obj

    for slot in obj.material_slots:
        matnodes = slot.material.node_tree.nodes
        tex = matnodes.new('ShaderNodeTexImage')
        teximg = glob(join(data_path, 'tex/*'))[0]
        tex.image = bpy.data.images.load(teximg)
        disp=slot.material.node_tree.nodes["Principled BSDF"].inputs['Base Color']
        slot.material.node_tree.links.new(disp, tex.outputs[0])


    # Possibly disable specular shading
    for slot in obj.material_slots:
        node = slot.material.node_tree.nodes['Principled BSDF']
        node.inputs['Specular'].default_value = 0.0


    # Set objekt IDs
    obj.pass_index = 1

    if add_light:
        bpy.ops.object.select_by_type(type='LIGHT')
        bpy.ops.object.delete()

    # Place camera
    if prev is None:
        cam = scene.objects['Camera']
        cam.location = (0, 1.5, 0)
        cam.data.lens = 35
        cam.data.sensor_width = 50
        cam.data.type = 'ORTHO'
        cam.data.ortho_scale = 2

        cam_constraint = cam.constraints.new(type='TRACK_TO')
        cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
        cam_constraint.up_axis = 'UP_Y'

        cam_empty = bpy.data.objects.new("Empty", None)
        cam_empty.location = (0, 0, 0)
        cam.parent = cam_empty

        scene.collection.objects.link(cam_empty)
        context.view_layer.objects.active = cam_empty
        cam_constraint.target = cam_empty

        bpy.context.view_layer.update()
    else:
        cam_empty = scene.objects["Empty"] 
        cam_empty.rotation_euler[2] = 0

        cam = scene.objects['Camera']


    objname = glob(join(data_path, 'rp*.obj'))[0]

    step = 20
    # cam_empty.rotation_euler[2] += math.radians(180)

    for i in range(step):
        imgname = objname.split('/')[-1].split('.')[0] + str(i)
        scene.render.filepath = join(join(out_path, 'image'), imgname)
        normal_file_output.file_slots[0].path = join(join(out_path, 'normal'), imgname)
        # albedo_file_output.file_slots[0].path = join(join(out_path, 'albedo'), imgname)

        bpy.ops.render.render(write_still=True)

        R_bcam2cv = Matrix(((1, 0,  0),(0, -1, 0),(0, 0, -1)))
        location, rotation = cam.matrix_world.decompose()[0:2]
        R_world2bcam = rotation.to_matrix().transposed()
        T_world2bcam = -1*R_world2bcam @ location
        R_world2cv = R_bcam2cv@R_world2bcam
        T_world2cv = R_bcam2cv@T_world2bcam
        calibration = np.eye(4, dtype=np.float32)
        for k in range(3):
            for j in range(3):
                calibration[k,j] = R_world2cv[k][j]
        for j in range(3):
            calibration[j,3] = T_world2cv[j]


        transform = np.empty((2, 3), dtype=np.float32)

        camd = cam.data
        f_in_mm = camd.lens
        if cam.data.type == 'ORTHO':
            depsgraph = context.evaluated_depsgraph_get()
            projection_matrix = cam.calc_matrix_camera(depsgraph, x = render.resolution_x, y = render.resolution_y)
            transform[0,0] = projection_matrix[0][0] * 512
            transform[1,1] = projection_matrix[1][1] * 512
            transform[0,2] = projection_matrix[0][2] * 512 + 512
            transform[1,2] = projection_matrix[1][2] * 512 + 512
        else:
            scale = 1.
            resolution_x_in_px = scale * render.resolution_x
            resolution_y_in_px = scale * render.resolution_y
            sensor_size_in_mm = get_sensor_size(camd.sensor_fit, camd.sensor_width, camd.sensor_height)
            sensor_fit = get_sensor_fit(
                camd.sensor_fit,
                scene.render.pixel_aspect_x * resolution_x_in_px,
                scene.render.pixel_aspect_y * resolution_y_in_px
            )
            pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
            if sensor_fit == 'HORIZONTAL':
                view_fac_in_px = resolution_x_in_px
            else:
                view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px
            pixel_size_mm_per_px = sensor_size_in_mm / f_in_mm / view_fac_in_px
            s_u = 1 / pixel_size_mm_per_px
            s_v = 1 / pixel_size_mm_per_px / pixel_aspect_ratio
            u_0 = resolution_x_in_px / 2 - camd.shift_x * view_fac_in_px
            v_0 = resolution_y_in_px / 2 + camd.shift_y * view_fac_in_px / pixel_aspect_ratio
            skew = 0 # only use rectangular pixels
            
            transform[0, 0] = s_u
            transform[0, 1] = skew
            transform[0, 2] = u_0
            transform[1, 0] = 0
            transform[1, 1] = s_v
            transform[1, 2] = v_0


        tmp = np.zeros([4,4], dtype=np.float32)
        tmp[0,0] = 1
        tmp[1,2] = -1
        tmp[2,1] = 1
        tmp[3,3] = 1
        calib = np.matmul(calibration, tmp)

        camera = dict()
        camera['calib'] = calib
        camera['transform'] = transform

        np.save(join(join(out_path, 'camera'), imgname+'.npy'), camera)

        cam_empty.rotation_euler[2] += math.radians(360/step)

        bpy.context.view_layer.update()

        for obj in bpy.context.selected_objects:
            bpy.context.scene.objects.active = obj
            bpy.context.object.cycles_visibility.shadow = False


    return obj

# blender --background --python blender_render.py

if __name__ == '__main__':
    dires = glob('demo_data/mesh/rp_1_3_OBJ')
    count = 0
    prev = None
    for dire in dires:
        prev = render(dire, 'demo_data', prev)
        count += 1
