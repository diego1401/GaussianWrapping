bl_info = {
    "name": "Mesh-based Editing Tool for MILo",
    "author": "Antoine Guedon",
    "blender": (4, 0, 2),
    "description": "Can be used to import and edit meshes reconstructed with MILo, as well as to render edited scenes with Gaussians.",
}

import os
import json
import numpy as np
import bpy
from bpy_extras.io_utils import ExportHelper, ImportHelper


def get_mesh_vertex_idx(mesh):
    vert_idx_values = np.zeros(len(mesh.vertices), dtype=int)
    mesh.attributes['index'].data.foreach_get("value", vert_idx_values)
    return vert_idx_values


def get_mesh_vertex_xyz(mesh):
    vert_xyz = np.zeros((len(mesh.vertices), 3)).flatten()
    mesh.attributes['position'].data.foreach_get("vector", vert_xyz)
    return vert_xyz.reshape(-1, 3)


def get_mesh_vertex_metadata(mesh):
    vert_idx_values = np.zeros(len(mesh.vertices), dtype=int)
    mesh.attributes['metadata'].data.foreach_get("value", vert_idx_values)
    return vert_idx_values


def get_text(text_data):
    return text_data.body


def set_text(text_data, new_text:str):
    text_data.body = new_text
    

def get_gs_metadata(metadata_name:str):
    metadata_object = bpy.data.objects[metadata_name]
    metadata = {}
    for child in metadata_object.children:
        metadata_text = get_text(child.data)
        metadata_dict = {}
        metadata_parsed_txt = metadata_text.split(';')
        for parsed_txt in metadata_parsed_txt:
            metadata_dict[parsed_txt.split(':::')[0]] = parsed_txt.split(':::')[-1]
        metadata_idx = str(child.name)
        metadata[metadata_idx] = metadata_dict
    return metadata


def is_gs_mesh(mesh):
    return ('metadata' in mesh.attributes)


def is_windows_path(path:str):
    if '\\' in path:
        return True
    
    
def convert_path_to_linux(path:str):
    return path.replace('\\\\', '\\').replace('\\', '/')


def get_matrix_world(obj):
    return [[obj.matrix_world[i][j] for j in range(4)] for i in range(4)]


class QueryProps(bpy.types.PropertyGroup):
    n_checkpoints: bpy.props.IntProperty(name="Number of checkpoints", default=5, min=1, max=10)
    
    gs_dir: bpy.props.StringProperty(
        name="Path to MILo directory", 
        default="./milo",
        description="Path to the directory cloned from MILo's repo",
    )
    
    mesh_file_to_load: bpy.props.StringProperty(
        name="Path to mesh PLY", 
        default="",
        description="Path to the mesh file (in PLY format) to load",
    )
    
    checkpoint_to_load: bpy.props.StringProperty(
        name="Path to 3DGS PLY", 
        default="",
        description="Path to the MILo 3DGS file (in PLY format) to use for rendering",
    )
    

class WMgsSelector(bpy.types.Operator, ImportHelper):
    """Select GS directory"""
    bl_idname = "something.gs_selector"
    bl_label = "Select GS folder"

    filename: bpy.props.StringProperty()
    filter_glob: bpy.props.StringProperty(
        default="",
        options={'HIDDEN'},
    )

    def execute(self, context):
        fdir = self.properties.filepath
        bpy.context.scene.QueryProps.gs_dir = fdir
        return{'FINISHED'}
    
    def invoke(self, context, event):
        self.filename = ""
        wm = context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
    
class WMgsMeshSelector(bpy.types.Operator, ImportHelper):
    """Select mesh file"""
    bl_idname = "something.mesh_selector"
    bl_label = "Select mesh file"

    filename_ext = ".ply"
    filter_glob: bpy.props.StringProperty(
        default="*.ply",
        options={'HIDDEN'},
    )

    def execute(self, context):
        fdir = self.properties.filepath
        bpy.context.scene.QueryProps.mesh_file_to_load = fdir
        return{'FINISHED'}
    

class WMgsCheckpointSelector(bpy.types.Operator, ImportHelper):
    """Select checkpoint file"""
    bl_idname = "something.checkpoint_selector"
    bl_label = "Select checkpoint file"

    filename_ext = ".ply"
    filter_glob: bpy.props.StringProperty(
        default="*.ply",
        options={'HIDDEN'},
    )

    def execute(self, context):
        fdir = self.properties.filepath
        bpy.context.scene.QueryProps.checkpoint_to_load = fdir
        return{'FINISHED'}


def create_render_package(
    query_props,
    gs_metadata,
    start_frame,
    end_frame,
    just_render_current_screen=False,
):
    # Camera object
    camera_data = {}
    camera_data['matrix_world'] = []
    camera_data['lens'] = []
    camera_data['angle'] = []
    camera_data['angle_x'] = []
    camera_data['angle_y'] = []
    camera_data['clip_start'] = []
    camera_data['clip_end'] = []
    cam_obj = bpy.context.scene.camera
    
    camera_data['image_width'] = bpy.context.scene.render.resolution_x
    camera_data['image_height'] = bpy.context.scene.render.resolution_y
    print(f"Output images have resolution: {camera_data['image_width']} x {camera_data['image_height']}")
    
    # Create Meshes data
    meshes_data = []
    n_mesh_owners = 0
    n_posable_mesh_owners = 0
    mesh_owners = []
    posable_mesh_owners = []
    for ob in bpy.data.objects:
        if (ob.type == 'MESH') and (not ob.hide_render) and is_gs_mesh(ob.data):
            n_mesh_owners += 1
            mesh_owners.append(ob)
            if ob.parent and ob.parent.type=='ARMATURE':
                n_posable_mesh_owners += 1
                posable_mesh_owners.append(ob)
    plural_print = '' if n_mesh_owners<=1 else 'es'
    print(f"\n{n_mesh_owners} GS mesh{plural_print} detected in the scene,")
    plural_print = '' if n_posable_mesh_owners<=1 else 'es'
    print(f"including {n_posable_mesh_owners} posable GS mesh{plural_print}.")        
    
    for i_mesh in range(n_mesh_owners):
        mesh = mesh_owners[i_mesh].data
        mesh_metadata = get_mesh_vertex_metadata(mesh)
        mesh_metadata_dict = gs_metadata[str(mesh_metadata[0])]
        mesh_name = mesh_metadata_dict['mesh_name']
        checkpoint_name = mesh_metadata_dict['checkpoint_name']
        mesh_data = {
            'mesh_name': mesh_name,
            'checkpoint_name': checkpoint_name,
            'matrix_world': np.array(mesh_owners[i_mesh].matrix_world).tolist(),
            'xyz': get_mesh_vertex_xyz(mesh).tolist(),
            'idx': get_mesh_vertex_idx(mesh).tolist(),
            'metadata': get_mesh_vertex_metadata(mesh).tolist(),
        }
        meshes_data.append(mesh_data)
    
    # Create rest bones data
    bones_data = []
    for i_mesh in range(n_mesh_owners):
        mesh_obj = mesh_owners[i_mesh]
        mesh = mesh_obj.data
        vertices = mesh.vertices
        if mesh_obj in posable_mesh_owners:
            armature_obj = mesh_obj.parent
            armature = armature_obj.data
            armature.pose_position = 'REST'
            
            # Armature data
            armature_dict = {}
            armature_dict['matrix_world'] = [[armature_obj.matrix_world[i][j] for j in range(4)] for i in range(4)]
            armature_dict['rest_bones'] = {}
            armature_dict['pose_bones'] = {}
            for bone in armature.bones:
                mat = armature_obj.matrix_world @ bone.matrix_local
                mat_list = [[mat[i][j] for j in range(4)] for i in range(4)]
                armature_dict['rest_bones'][bone.name] = mat_list
                armature_dict['pose_bones'][bone.name] = []
            
            # Vertex group data
            vertex_dict = {}
            vertex_dict['matrix_world'] = get_matrix_world(mesh_obj)
            vertex_dict['tpose_points'] = []
            vertex_dict['groups'] = []
            vertex_dict['weights'] = []
            vertex_group_names = {}
            for i in range(len(mesh_obj.vertex_groups)):
                group = mesh_obj.vertex_groups[i]
                vertex_group_names[str(group.index)] = group.name
            for i in range(len(vertices)):
                v = mesh_obj.matrix_world @ vertices[i].co
                vertex_dict['tpose_points'].append([v[0], v[1], v[2]])
                group_list = []
                weight_list = []
                for group in vertices[i].groups:
                    group_list.append(vertex_group_names[str(group.group)])
                    weight_list.append(group.weight)
                vertex_dict['groups'].append(group_list)
                vertex_dict['weights'].append(weight_list)
                
            armature.pose_position = 'POSE'
            
            bones_dict = {
                "armature": armature_dict,
                "vertex": vertex_dict
            }
        else:
            bones_dict = None
        bones_data.append(bones_dict)
    
    if just_render_current_screen:
        start_frame, end_frame = 1, 1
    
    for i_frame in range(start_frame, end_frame+1):
        # Set frame
        if not just_render_current_screen:
            bpy.context.scene.frame_set(i_frame)
        
        # Create camera data
        camera_data['matrix_world'].append([[cam_obj.matrix_world[i][j] for j in range(4)] for i in range(4)])
        camera_data['lens'].append(cam_obj.data.lens)
        camera_data['angle'].append(cam_obj.data.angle)
        camera_data['angle_x'].append(cam_obj.data.angle_x)
        camera_data['angle_y'].append(cam_obj.data.angle_y)
        camera_data['clip_start'].append(cam_obj.data.clip_start)
        camera_data['clip_end'].append(cam_obj.data.clip_end)
        
        # Create Bones data
        for i_mesh in range(n_mesh_owners):
            mesh_obj = mesh_owners[i_mesh]
            if mesh_obj in posable_mesh_owners:
                armature_obj = mesh_obj.parent
                pose = armature_obj.pose
                armature = armature_obj.data
                for bone in pose.bones:
                    mat = armature_obj.matrix_world @ bone.matrix
                    mat_list = [[mat[i][j] for j in range(4)] for i in range(4)]
                    bones_data[i_mesh]['armature']['pose_bones'][bone.name].append(mat_list)
    
    # Build and save render package
    # TODO
    render_package = {
        'camera': camera_data,
        'meshes': meshes_data,
        'bones': bones_data
    }
    
    return render_package


class RendergsOperator(bpy.types.Operator):
    """Render using GS"""
    bl_idname = "object.gs_render"
    bl_label = "GS Renderer"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        query_props = bpy.context.scene.QueryProps
        print("\nStart rendering with GS...")
        print("---> GSdirectory:", query_props.gs_dir)
        
        gs_metadata_name = "GS metadata (do not delete)"
        gs_metadata = get_gs_metadata(gs_metadata_name)
        print("\nGS metadata:", gs_metadata)
        
        scene = context.scene
        cursor = scene.cursor.location
        obj = context.active_object
        
        start_frame = bpy.context.scene.frame_start
        end_frame = bpy.context.scene.frame_end
        print("\nStart frame:", start_frame)
        print("End frame:", end_frame)
        
        # Output path (TO CHECK)
        scene_name = os.path.splitext(os.path.basename(bpy.data.filepath))[0] + '.json'
        package_output_dir = os.path.join(query_props.gs_dir, 'blender', 'packages')
        os.makedirs(package_output_dir, exist_ok=True)
        package_output_file_path = os.path.join(package_output_dir, scene_name)

        render_package = create_render_package(
            query_props,
            gs_metadata,
            start_frame,
            end_frame,
            just_render_current_screen=False,
        )
        
        with open(package_output_file_path, "w") as outfile:
            json.dump(render_package, outfile)
            print(f'Results saved to "{package_output_file_path}".')

        return {'FINISHED'}
    
    
class RendergsOperatorSingleImage(bpy.types.Operator):
    """Render using GS"""
    bl_idname = "object.gs_render_single"
    bl_label = "GS Renderer"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        query_props = bpy.context.scene.QueryProps
        print("\nStart rendering with GS...")
        print("---> GSdirectory:", query_props.gs_dir)
        
        gs_metadata_name = "GS metadata (do not delete)"
        gs_metadata = get_gs_metadata(gs_metadata_name)
        print("\nGS metadata:", gs_metadata)
        
        scene = context.scene
        cursor = scene.cursor.location
        obj = context.active_object
        
        start_frame = bpy.context.scene.frame_start
        end_frame = bpy.context.scene.frame_end
        print("\nStart frame:", start_frame)
        print("End frame:", end_frame)
        
        # Output path (TO CHECK)
        scene_name = os.path.splitext(os.path.basename(bpy.data.filepath))[0] + '.json'
        package_output_dir = os.path.join(query_props.gs_dir, 'blender', 'packages')
        os.makedirs(package_output_dir, exist_ok=True)
        package_output_file_path = os.path.join(package_output_dir, scene_name)

        render_package = create_render_package(
            query_props,
            gs_metadata,
            start_frame,
            end_frame,
            just_render_current_screen=True,
        )
        
        with open(package_output_file_path, "w") as outfile:
            json.dump(render_package, outfile)
            print(f'Results saved to "{package_output_file_path}".')

        return {'FINISHED'}
    
    
class AddgsMeshOperator(bpy.types.Operator):
    """Add a mesh reconstructed with GS to the scene"""
    bl_idname = "object.add_gs_mesh"
    bl_label = "Add MILo Mesh"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        gs_metadata_name = "GS metadata (do not delete)"
        
        scene = context.scene
        cursor = scene.cursor.location
        obj = context.active_object
        
        # Create metadata object if needed
        if gs_metadata_name in bpy.data.objects:
            metadata_object = bpy.data.objects[gs_metadata_name]
        else:
            bpy.ops.object.empty_add(
                type='PLAIN_AXES', 
                align='WORLD', 
                location=(0, 0, 0), 
                scale=(1, 1, 1)
            )
            metadata_object = bpy.context.active_object
            metadata_object.name = gs_metadata_name
            metadata_object.hide_viewport = True
            metadata_object.hide_render = True
        
        query_props = bpy.context.scene.QueryProps
        
        print("\nStart loading GS mesh...")
        print("---> Mesh to load:", query_props.mesh_file_to_load)
        
        # ---Load mesh---
        bpy.ops.wm.ply_import(filepath=query_props.mesh_file_to_load)
        obj = bpy.context.selected_objects[-1]
        
        # ---Set Rotation for better visualization---
        obj.rotation_euler[0] = -np.pi / 2
        
        # ---Material---
        # Set up
        # material = obj.active_material
        material = bpy.data.materials.new(obj.name)
        material.use_nodes = True
        nodes = material.node_tree.nodes
        p_bsdf_inputs = nodes['Principled BSDF'].inputs
        
        # Build new Base Color node
        rgb_node = nodes.new('ShaderNodeRGB')
        rgb_node.outputs[0].default_value = (0., 0., 0., 1.)
        
        # Set up new Emission Color node
        vertex_color_node = nodes.new(type="ShaderNodeVertexColor")
        vertex_color_node.layer_name = "Col"
        
        # Links nodes
        material.node_tree.links.new(rgb_node.outputs[0], p_bsdf_inputs['Base Color'])
        material.node_tree.links.new(vertex_color_node.outputs[0], p_bsdf_inputs['Emission Color'])
        p_bsdf_inputs['Emission Strength'].default_value = 1.
        
        # Set material
        obj.data.materials.append(material)
        
        # Removes specularity and roughness
        p_bsdf_inputs['Roughness'].default_value = 0.0
        p_bsdf_inputs['IOR'].default_value = 1.0

        # ---Create index data---
        mesh = obj.data
        vert_idx_values = np.arange(len(mesh.vertices)).tolist()        
        vert_idx_attribute = mesh.attributes.new(name="index", type="INT", domain="POINT")
        vert_idx_attribute.data.foreach_set("value", vert_idx_values)
        
        # ---Write metadata---
        if True:
            mesh_name = convert_path_to_linux(query_props.mesh_file_to_load)
            checkpoint_name = convert_path_to_linux(query_props.checkpoint_to_load)
            metadata_string = ''
            metadata_string = metadata_string + 'mesh_name:::' + mesh_name 
            metadata_string = metadata_string + ';checkpoint_name:::' + checkpoint_name
            
            max_idx = -1
            for child in metadata_object.children:
                tmp_idx = int(child.name)
                if tmp_idx > max_idx:
                    max_idx = tmp_idx
            new_idx = max_idx + 1
            
            # Text object
            bpy.ops.object.text_add(enter_editmode=False, align='WORLD')
            text_obj = bpy.context.active_object
            text_obj.name = str(new_idx)
            set_text(text_obj.data, metadata_string)
            text_obj.hide_render = True
            text_obj.hide_viewport = True
            text_obj.parent = metadata_object
            
            # Give to every vertex the corresponding metadata idx
            vert_idx_values = (new_idx + np.zeros(len(mesh.vertices), dtype=int)).tolist()
            vert_idx_attribute = mesh.attributes.new(name="metadata", type="INT", domain="POINT")
            vert_idx_attribute.data.foreach_set("value", vert_idx_values)
            
        # ---Set the viewport shading for better visualization---
        area = next(area for area in bpy.context.screen.areas if area.type == 'VIEW_3D')
        space = next(space for space in area.spaces if space.type == 'VIEW_3D')
        space.shading.type = 'RENDERED'
        
        return {'FINISHED'}
    
    
class AddgsMeshPanel(bpy.types.Panel):
    """ Display panel in 3D view"""
    bl_label = "Add MILo mesh"

    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'render'
    
    def draw(self, context):
        layout = self.layout
        
        # Layout
        row = layout.row(align=True)
        row.alignment = 'EXPAND'
        
        text_col = row.column(align=True)
        text_col_row0 = text_col.row(align=True)
        text_col_row0.alignment = 'RIGHT'
        text_col_row1 = text_col.row(align=True)
        text_col_row1.alignment = 'RIGHT'
        
        field_col = row.column(align=True)
        field_col_row0 = field_col.row(align=True)
        field_col_row1 = field_col.row(align=True)
        
        main_button = layout.column(align=True)
        
        # Props
        query_props = bpy.context.scene.QueryProps
        
        # Select mesh
        text_col_row0.label(text='Path to mesh PLY   ')
        field_col_row0.prop(query_props, 'mesh_file_to_load', text='')
        field_col_row0.operator("something.mesh_selector", icon="FILE_FOLDER", text="")
        
        # Select checkpoint
        text_col_row1.label(text='Path to 3DGS PLY   ')
        field_col_row1.prop(query_props, 'checkpoint_to_load', text='')
        field_col_row1.operator("something.checkpoint_selector", icon="FILE_FOLDER", text="")
        
        # Add mesh
        props = main_button.operator("object.add_gs_mesh", 
            text='Add mesh',
            emboss=True,
            icon="MESH_CUBE"
        )
    
    
class RendergsPanel(bpy.types.Panel):
    """ Display panel in 3D view"""
    bl_label = "Render MILo scene"
    
    if True:
        bl_space_type = 'PROPERTIES'
        bl_region_type = 'WINDOW'
        bl_context = 'render'
    else:
        bl_region_type = "UI"
        bl_space_type = "VIEW_3D"
    
        bl_options = {'HEADER_LAYOUT_EXPAND'}
    
    def draw(self, context):
        layout = self.layout
        
        # Parameters        
        row = layout.row(align=True)
        row.alignment = 'EXPAND'
        
        if False:
            empty_col = row.column(align=False)
            empty_col.label(text='')
            empty_col.scale_x = 0.2
        
        text_col = row.column(align=True)
        text_col_row0 = text_col.row(align=True)
        text_col_row1 = text_col.row(align=True)
        text_col_row2 = text_col.row(align=True)
        text_col_row0.alignment = 'RIGHT'
        text_col_row1.alignment = 'RIGHT'
        text_col_row2.alignment = 'RIGHT'
        
        field_col = row.column(align=True)
        field_col_row0 = field_col.row(align=True)
        field_col_row1 = field_col.row(align=True)
        field_col_row2 = field_col.row(align=True)
        
        # Main button
        main_button = layout.column(align=True)
        main_button2 = layout.column(align=True)
        
        # Props
        query_props = bpy.context.scene.QueryProps
        props = main_button.operator("object.gs_render_single", 
            text='Render Image',
            emboss=True,
            icon="RENDER_STILL"
        )
        
        props2 = main_button2.operator("object.gs_render", 
            text='Render Animation',
            emboss=True,
            icon="RENDER_ANIMATION"
        )
        
        text_col_row1.label(text='Path to MILo directory   ')
        
        field_col_row1.prop(query_props, 'gs_dir', text='')
        field_col_row1.operator("something.gs_selector", icon="FILE_FOLDER", text="")
    
    
def menu_func(self, context):
    self.layout.operator(RendergsOperator.bl_idname)


classes = (
    QueryProps,
    WMgsSelector,
    WMgsMeshSelector,
    WMgsCheckpointSelector,
    AddgsMeshPanel,
    AddgsMeshOperator,
    RendergsPanel,
    RendergsOperator,
    RendergsOperatorSingleImage,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.QueryProps = bpy.props.PointerProperty(type=QueryProps)


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
    del(bpy.types.Scene.QueryProps)


if __name__ == "__main__":
    register()