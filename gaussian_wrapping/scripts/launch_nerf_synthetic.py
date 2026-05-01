import os
import sys
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)


scenes = [
    "chair",
    "drums",
    "ficus",
    "hotdog",
    "lego",
    "materials",
    "mic",
    "ship",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("NeRF Synthetic benchmark: train + extract mesh (radegs rasterizer).\n\n"
                     "Note: We only launch the training and extraction with the RaDe-GS rasterizer "
                     "since it leads to smoother meshes. There are no metrics in this script. "
                     "We provide it just as an example of our method being applied to synthetic scenes."),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory containing one subdirectory per NeRF Synthetic scene.")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Where to save trained models and meshes.")
    parser.add_argument("--gpu_device", type=str, default="0",
                        help="CUDA device index.")
    parser.add_argument("--data_on_gpu", action="store_true",
                        help="Load dataset on GPU instead of CPU.")
    parser.add_argument("--depth_order", action="store_true",
                        help="Enable depth-order regularization using a pre-trained monocular depth model. "
                             "Not used in the paper; can improve results at the cost of requiring monocular depth priors.")
    parser.add_argument("--depth_order_config", type=str, default=None,
                        help="Depth-order config name (see configs/depth_order/). "
                             "Only used when --depth_order is set.")
    args = parser.parse_args()

    data_device = "cuda" if args.data_on_gpu else "cpu"

    for scene_name in scenes:
        print(f"\n[INFO] ===== {scene_name} =====")

        scene_data_dir = os.path.join(args.data_dir, scene_name)
        scene_output_dir = os.path.join(args.output_dir, scene_name)

        train_extract_command = " ".join(filter(None, [
            f"CUDA_VISIBLE_DEVICES={args.gpu_device}",
            "python gaussian_wrapping/scripts/train_and_extract_gw_radegs.py",
            f"-s {scene_data_dir}",
            f"-m {scene_output_dir}",
            f"--data_device {data_device}",
            "--depth_order" if args.depth_order else "",
            "--random_background",
            "--isosurface_value 0.2",
            "--log_interval 1000",
            "--no_postprocess",
            f"--depth_order_config {args.depth_order_config}" if args.depth_order and args.depth_order_config else "",
        ]))

        print("\n[INFO] Running train & extract command:", train_extract_command, sep="\n")
        os.system(train_extract_command)

        # Run PAM Extraction
        input_mesh = os.path.join(scene_output_dir, "mesh_exact_computation_2pivots_transmittance_threshold_0.7_searched.ply")
        pam_output_mesh = os.path.join(scene_output_dir, f"{scene_name}_pam.ply")
        
        pam_extract_command = " ".join(filter(None, [
            f"CUDA_VISIBLE_DEVICES={args.gpu_device}",
            "python gaussian_wrapping/primal_adaptive_meshing_extraction.py",
            f"-s {scene_data_dir}",
            f"-m {scene_output_dir}",
            f"--input_mesh {input_mesh}",
            f"--output_mesh {pam_output_mesh}",
            "--max_points 2000000",
            "--bounding_box_method scene",
            "--iso_surface_value 0.2",
        ]))

        print("\n[INFO] Running PAM extraction command:", pam_extract_command, sep="\n")
        os.system(pam_extract_command)

        # Run Texture Refinement on the PAM mesh
        texture_command = " ".join(filter(None, [
            f"CUDA_VISIBLE_DEVICES={args.gpu_device}",
            "python gaussian_wrapping/texture_mesh.py",
            f"-s {scene_data_dir}",
            f"-m {scene_output_dir}",
            f"--mesh {pam_output_mesh}",
            "--n_iter 1000",
            "--lambda_dssim 0.2",
            "--lr 0.0025",
            "--sh_degree_for_texturing 0",
            "--use_scalable_renderer"
        ]))

        print("\n[INFO] Running Texture Refinement on PAM mesh:", texture_command, sep="\n")
        os.system(texture_command)
