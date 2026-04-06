from argparse import ArgumentParser
import os
import sys
import bpy


def get_args():
    parser = ArgumentParser(description="Decimate a mesh using Blender.")
    parser.add_argument("--in", dest="in_path", required=True, help="Input PLY path (e.g., ./output/Ignatius/mesh_learnable_sdf.ply)")
    parser.add_argument("--out", dest="out_path", default=None, help="Output PLY path (e.g., ./output/Ignatius/mesh_learnable_sdf_decimated.ply)")
    parser.add_argument("--ratio", type=float, default=0.3, help="Decimation ratio (default: 0.3)")

    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    
    return parser.parse_args(argv)


if __name__ == '__main__':    
    args = get_args()
    
    MODIFIER_NAME = 'DecimateMod'
    
    # Output path
    if args.out_path is None:
        out_path = args.in_path.replace('.ply', '_decimated_with_blender.ply')
    
    # Create output dir
    decimated_dir = os.path.dirname(out_path)
    if not os.path.exists(decimated_dir):
        os.makedirs(decimated_dir)

    # Clear Blender scene
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)

    # Import mesh
    print('[INFO] Importing mesh from path:', args.in_path)
    bpy.ops.wm.ply_import(filepath=args.in_path)

    # Decimate mesh
    print(f'[INFO] Starting decimation with ratio {args.ratio}')
    modifier = bpy.context.object.modifiers.new(MODIFIER_NAME, 'DECIMATE')
    modifier.ratio = args.ratio
    bpy.ops.object.modifier_apply(modifier=MODIFIER_NAME)
    print('[INFO] Decimation is done.')

    # Export mesh
    print('[INFO] Exporting mesh...')
    bpy.ops.wm.ply_export(filepath=out_path)
    print('[INFO] Mesh exported.')
