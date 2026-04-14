import subprocess
import sys
import os
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_SCRIPT = os.path.join(BASE_DIR, "train.py")
EXTRACT_SCRIPT = os.path.join(BASE_DIR, "pivot_based_mesh_extraction.py")
TEXTURE_SCRIPT = os.path.join(BASE_DIR, "texture_mesh.py")


TRAIN_FLAGS = [
    "--rasterizer", "radegs",
    "--regularization_from_iter", "15000",
    "--multiview_config", "fast_late",
    "--multiview_factor", "0.05",
    "--use_max_size_threshold",
    "--data_device", "cpu",
    "--N_max_gaussians", "6000000"
]

EXTRACT_FLAGS = [
    "--sdf_mode", "exact_computation",
    "--rasterizer", "radegs",
    "--dtype", "int32",
    # isosurface_value appended dynamically
    "--n_binary_steps", "10",
    "--iteration", "30000",
    "--use_valid_mask",
    "--postprocess",
    "--std_factor", "3.33",
    "--use_searched_pivots",
    "--search_iter", "5",
    "--search_step_size", "0.33",
    "--data_device", "cpu"
]

def parse_data_args(args):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-s", "--source_path")
    parser.add_argument("-m", "--model_path")
    parser.add_argument("-r", "--resolution")
    known, _ = parser.parse_known_args(args)
    result = []
    if known.source_path: result += ["-s", known.source_path]
    if known.model_path:  result += ["-m", known.model_path]
    if known.resolution:  result += ["-r", known.resolution]
    return result, known

# Intercept extraction-only parameters from sys.argv 
parser_ext = argparse.ArgumentParser(add_help=False)
parser_ext.add_argument("--isosurface_value", type=float, default=0.0)
parser_ext.add_argument("--no_postprocess", action="store_true", help="Disable the postprocessing step.")
parser_ext.add_argument("--apply_decimation", action="store_true", help="Apply Blender decimation before texturing.")
parser_ext.add_argument("--decimate_ratio", type=float, default=0.3, help="Decimation ratio for Blender.")
ext_args, train_args = parser_ext.parse_known_args(sys.argv[1:])

if ext_args.no_postprocess and "--postprocess" in EXTRACT_FLAGS:
    EXTRACT_FLAGS.remove("--postprocess")

# Add intercepted args to extract flags
EXTRACT_FLAGS.extend(["--isosurface_value", str(ext_args.isosurface_value)])

shared_args, known = parse_data_args(train_args)

print("[INFO] Step 1/3: Training...")
result = subprocess.run([sys.executable, TRAIN_SCRIPT] + TRAIN_FLAGS + train_args)
if result.returncode != 0:
    print("[ERROR] Training failed. Aborting extraction.")
    sys.exit(result.returncode)

print("[INFO] Step 2/3: Extracting mesh...")
result = subprocess.run([sys.executable, EXTRACT_SCRIPT] + EXTRACT_FLAGS + shared_args)
if result.returncode != 0:
    print("[ERROR] Mesh extraction failed. Aborting texture refinement.")
    sys.exit(result.returncode)

# Dynamically determine the mesh name based on args
sdf_mode = "exact_computation"
if "--sdf_mode" in EXTRACT_FLAGS:
    sdf_mode = EXTRACT_FLAGS[EXTRACT_FLAGS.index("--sdf_mode") + 1]

if sdf_mode == "exact_computation":
    transmittance_threshold = 0.5 + ext_args.isosurface_value
    iso_suffix = f"_transmittance_threshold_{transmittance_threshold}" if transmittance_threshold != 0.5 else ""
else:
    iso_suffix = f"_iso_{ext_args.isosurface_value}" if ext_args.isosurface_value != 0.0 else ""

MESH_NAME = f"mesh_{sdf_mode}_2pivots{iso_suffix}_searched.ply"
if "--postprocess" in EXTRACT_FLAGS:
    MESH_NAME = MESH_NAME.replace(".ply", "_post.ply")

mesh_path = os.path.join(known.model_path, MESH_NAME) if known.model_path else MESH_NAME

if ext_args.apply_decimation:
    print(f"[INFO] Step 2.5: Decimating mesh with ratio {ext_args.decimate_ratio}...")
    blender_script = os.path.join(BASE_DIR, "mesh_decimate.py")
    result = subprocess.run([
        "blender", "-b", "-P", blender_script, "--",
        "--in", mesh_path, "--ratio", str(ext_args.decimate_ratio)
    ])
    if result.returncode != 0:
        print("[ERROR] Decimation failed. Aborting texture refinement.")
        sys.exit(result.returncode)
    mesh_path = mesh_path.replace(".ply", "_decimated_with_blender.ply")

print("[INFO] Step 3/3: Refining texture...")
subprocess.run([sys.executable, TEXTURE_SCRIPT, "--rasterizer", "radegs", "--mesh", mesh_path] + shared_args)
