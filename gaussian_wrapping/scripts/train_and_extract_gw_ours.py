import subprocess
import sys
import os
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_SCRIPT = os.path.join(BASE_DIR, "train.py")
EXTRACT_SCRIPT = os.path.join(BASE_DIR, "pivot_based_mesh_extraction.py")
TEXTURE_SCRIPT = os.path.join(BASE_DIR, "texture_mesh.py")

MESH_NAME = "mesh_ours_2pivots_post.ply"

TRAIN_FLAGS = [
    "--rasterizer", "ours",
    "--feature_dc_lr", "0.0013",
    "--feature_rest_lr", "0.00011",
    "--exposure_compensation",
    "--data_device", "cpu",
    "--N_max_gaussians", "5000000"
]

EXTRACT_FLAGS = [
    "--sdf_mode", "ours",
    "--rasterizer", "ours",
    "--dtype", "int32",
    "--isosurface_value", "0.0",
    "--n_binary_steps", "10",
    "--iteration", "30000",
    "--use_valid_mask",
    "--postprocess",
    "--filter_large_edges",
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

user_args = sys.argv[1:]
shared_args, known = parse_data_args(user_args)

print("[INFO] Step 1/3: Training...")
result = subprocess.run([sys.executable, TRAIN_SCRIPT] + TRAIN_FLAGS + user_args)
if result.returncode != 0:
    print("[ERROR] Training failed. Aborting extraction.")
    sys.exit(result.returncode)

print("[INFO] Step 2/3: Extracting mesh...")
result = subprocess.run([sys.executable, EXTRACT_SCRIPT] + EXTRACT_FLAGS + shared_args)
if result.returncode != 0:
    print("[ERROR] Mesh extraction failed. Aborting texture refinement.")
    sys.exit(result.returncode)

mesh_path = os.path.join(known.model_path, MESH_NAME) if known.model_path else MESH_NAME

print("[INFO] Step 3/3: Refining texture...")
subprocess.run([sys.executable, TEXTURE_SCRIPT, "--rasterizer", "ours", "--mesh", mesh_path] + shared_args)
