import os
import sys
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)


scenes = [
    "bicycle",
    "bonsai",
    "counter",
    "garden",
    "kitchen",
    "room",
    "stump",
    "flowers",
    "treehill",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end MipNeRF 360 benchmark: train + extract + render + eval (radegs rasterizer)."
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory containing one subdirectory per MipNeRF 360 scene.")
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
    parser.add_argument("--apply_decimation", action="store_true",
                        help="Apply Blender decimation before texturing. NOTE: This willl not affect the mesh used for metric computation. This flag only outputs an additional decimated mesh.")
    parser.add_argument("--decimate_ratio", type=float, default=0.3,
                        help="Decimation ratio for Blender.")
    args = parser.parse_args()

    data_device = "cuda" if args.data_on_gpu else "cpu"

    for scene_name in scenes:
        print(f"\n[INFO] ===== {scene_name} =====")

        scene_data_dir = os.path.join(args.data_dir, scene_name)
        scene_output_dir = os.path.join(args.output_dir, scene_name)

        # ------------------------------------------------------------------ #
        # Step 1 – Train & Extract                                             #
        # ------------------------------------------------------------------ #
        train_extract_command = " ".join(filter(None, [
            f"CUDA_VISIBLE_DEVICES={args.gpu_device}",
            "python gaussian_wrapping/scripts/train_and_extract_gw_radegs.py",
            f"-s {scene_data_dir}",
            f"-m {scene_output_dir}",
            f"--data_device {data_device}",
            "--no-exposure_compensation",
            "--N_max_gaussians 6000000",
            "--depth_order" if args.depth_order else "",
            f"--depth_order_config {args.depth_order_config}" if args.depth_order and args.depth_order_config else "",
            "--apply_decimation" if args.apply_decimation else "",
            f"--decimate_ratio {args.decimate_ratio}" if args.apply_decimation else "",
        ]))

        print("\n[INFO] Running train & extract command:", train_extract_command, sep="\n")
        os.system(train_extract_command)

        # ------------------------------------------------------------------ #
        # Step 2 – Render                                                      #
        # ------------------------------------------------------------------ #
        render_command = " ".join([
            f"CUDA_VISIBLE_DEVICES={args.gpu_device}",
            "python gaussian_wrapping/render.py",
            f"-s {scene_data_dir}",
            f"-m {scene_output_dir}",
            "--rasterizer radegs",
            f"--data_device {data_device}",
        ])

        print("\n[INFO] Running render command:", render_command, sep="\n")
        os.system(render_command)

        # ------------------------------------------------------------------ #
        # Step 3 – Metrics                                                     #
        # ------------------------------------------------------------------ #
        metrics_command = " ".join([
            f"CUDA_VISIBLE_DEVICES={args.gpu_device}",
            "python gaussian_wrapping/metrics.py",
            f"-m {scene_output_dir}",
        ])

        print("\n[INFO] Running metrics command:", metrics_command, sep="\n")
        os.system(metrics_command)
