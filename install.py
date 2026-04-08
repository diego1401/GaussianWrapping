import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Setup the environment')
    
    parser.add_argument('--cuda_version', type=str, default='11.8', help='CUDA version to use', choices=['11.8', '12.1'])
    args = parser.parse_args()
    
    print(f"[INFO] Installing environment...")
    
    # Install the proper version of pip
    os.system(f"conda install -y pip=22.3.1")
    
    # Install torch
    print(f"[INFO] Installing torch...")
    os.system(f"conda install -y pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda={args.cuda_version} mkl=2023.1.0 -c pytorch -c nvidia")
    print(f"[INFO] Torch installed.")
    
    # Install requirements
    print(f"[INFO] Installing requirements...")
    os.system(f"pip install -r requirements.txt")
    print(f"[INFO] Requirements installed.")
    
    # Install submodules
    print(f"[INFO] Installing Mini-Splatting2 rasterizer...")
    os.system(f"pip install --no-build-isolation submodules/diff-gaussian-rasterization_ms")
    print("[INFO] Mini-Splatting2 rasterizer installed.")

    print(f"[INFO] Installing RaDe-GS rasterizer...")
    os.system(f"pip install --no-build-isolation submodules/diff-gaussian-rasterization")
    print("[INFO] RaDe-GS rasterizer installed.")

    print(f"[INFO] Installing Ours rasterizer...")
    os.system(f"pip install --no-build-isolation submodules/diff-gaussian-rasterization_ours")
    print("[INFO] Ours rasterizer installed.")

    print(f"[INFO] Installing SOF rasterizer...")
    os.system(f"pip install --no-build-isolation submodules/diff-gaussian-rasterization_sof")
    print("[INFO] SOF rasterizer installed.")
    
    print(f"[INFO] Installing Simple KNN...")
    os.system(f"pip install --no-build-isolation submodules/simple-knn")
    print("[INFO] Simple KNN installed.")
    
    print(f"[INFO] Installing Fused SSIM...")
    os.system(f"pip install --no-build-isolation submodules/fused-ssim")
    print("[INFO] Fused SSIM installed.")
    
    print(f"[INFO] Installing Triangulation...")
    os.chdir("submodules/tetra_triangulation/")
    os.system(f"conda install -y cmake")
    os.system(f"conda install -y conda-forge::gmp")
    os.system(f"conda install -y conda-forge::cgal")
    # WARNING: CUDA paths must be set before running cmake
    os.system(f"cmake . -DCMAKE_POLICY_VERSION_MINIMUM=3.5") # -DCMAKE_POLICY_VERSION_MINIMUM=3.5, Needed for RTX 4090
    os.system(f"make")
    os.system(f"pip install -e .")
    os.chdir("../../")
    print("[INFO] Triangulation installed.")
    
    print(f"[INFO] Installing Nvdiffrast...")
    os.chdir("submodules/nvdiffrast/")
    os.system(f"pip install --no-build-isolation -e .")
    os.chdir("../../")
    print("[INFO] Nvdiffrast installed.")

    print(f"[INFO] Installing Warp Patch NCC...")
    os.system(f"pip install --no-build-isolation submodules/Geometry-Grounded-Gaussian-Splatting/submodules/warp-patch-ncc")
    print("[INFO] Warp Patch NCC installed.")

    print(f"[INFO] Installing Torch Geometric...")
    os.system(f"pip install torch_geometric")
    cuda_tag = args.cuda_version.replace(".", "")
    os.system(f"pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu{cuda_tag}.html")
    print("[INFO] Torch Geometric installed.")

    print(f"[INFO] Installation complete.")
    