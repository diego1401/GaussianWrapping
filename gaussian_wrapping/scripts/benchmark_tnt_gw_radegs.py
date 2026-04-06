import os
import sys
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)


scenes_dict = {
    "Barn":        {"imp_metric": "outdoor"},
    "Caterpillar": {"imp_metric": "outdoor"},
    "Courthouse":  {"imp_metric": "outdoor"},
    "Ignatius":    {"imp_metric": "outdoor"},
    "Meetingroom": {"imp_metric": "indoor"},
    "Truck":       {"imp_metric": "outdoor"},
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end TNT benchmark: train + extract + eval (radegs rasterizer)."
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory containing one subdirectory per TNT scene.")
    parser.add_argument("--gt_dir", type=str, required=True,
                        help="Ground truth directory (Scene.ply, Scene.json, Scene_trans.txt, Scene_COLMAP_SfM.log).")
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

    for scene_name, scene_params in scenes_dict.items():
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
            f"--imp_metric {scene_params['imp_metric']}",
            f"--data_device {data_device}",
            "-r 2",
            "--depth_order" if args.depth_order else "",
            f"--depth_order_config {args.depth_order_config}" if args.depth_order and args.depth_order_config else "",
        ]))

        print("\n[INFO] Running train & extract command:", train_extract_command, sep="\n")
        os.system(train_extract_command)

        # ------------------------------------------------------------------ #
        # Step 2 – Locate the extracted mesh                                  #
        # ------------------------------------------------------------------ #
        mesh_path = os.path.join(scene_output_dir, "mesh_exact_computation_2pivots_searched_post.ply")
        if not os.path.exists(mesh_path):
            print(f"[WARNING] Mesh not found at {mesh_path}. "
                  "Skipping evaluations for this scene.")
            continue
        print(f"\n[INFO] Using mesh: {mesh_path}")

        traj_path = os.path.join(args.gt_dir, scene_name, f"{scene_name}_COLMAP_SfM.log")

        # ------------------------------------------------------------------ #
        # Step 3 – Uniform Sampling Evaluation                                #
        # ------------------------------------------------------------------ #
        uniform_eval_command = " ".join([
            "python gaussian_wrapping/eval/TNTUniScanEvals/uniform_sampling_eval.py",
            f"--dataset-dir {os.path.join(args.gt_dir, scene_name)}",
            f"--traj-path {traj_path}",
            f"--ply-path {mesh_path}",
            f"--out-dir {os.path.join(scene_output_dir, 'eval_uniform')}",
        ])

        print("\n[INFO] Running uniform sampling evaluation:", uniform_eval_command, sep="\n")
        os.system(uniform_eval_command)

        # ------------------------------------------------------------------ #
        # Step 4 – Virtual Scan Sampling Evaluation                           #
        # ------------------------------------------------------------------ #
        virtual_scan_eval_command = " ".join([
            "python gaussian_wrapping/eval/TNTUniScanEvals/virtual_scan_sampling_eval.py",
            f"-s {scene_data_dir}",
            "-r 2",
            "--data_device cpu",
            f"--dataset-dir {os.path.join(args.gt_dir, scene_name)}",
            f"--traj-path {traj_path}",
            f"--ply-path {mesh_path}",
            f"--out-dir {os.path.join(scene_output_dir, 'eval_virtual_scan')}",
        ])

        print("\n[INFO] Running virtual scan sampling evaluation:", virtual_scan_eval_command, sep="\n")
        os.system(virtual_scan_eval_command)

        # ------------------------------------------------------------------ #
        # Step 5 – Legacy TNT Evaluation                                      #
        # ------------------------------------------------------------------ #
        legacy_eval_command = " ".join([
            "python gaussian_wrapping/eval/tnt/run.py",
            f"--dataset-dir {os.path.join(args.gt_dir, scene_name)}",
            f"--traj-path {traj_path}",
            f"--ply-path {mesh_path}",
            f"--out-dir {os.path.join(scene_output_dir, 'eval_legacy')}",
        ])

        print("\n[INFO] Running legacy TNT evaluation:", legacy_eval_command, sep="\n")
        os.system(legacy_eval_command)
