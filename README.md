<div align="center">

<h1>From Blobs to Spokes: High-Fidelity Surface Reconstruction via Oriented Gaussians</h1>

<font size="4">
<a href="https://diego1401.github.io">Diego Gomez*<sup>1</sup></a>&emsp;
<a href="https://anttwo.github.io/">Antoine Guédon*<sup>1</sup></a>&emsp;
<a href="https://nissmar.github.io/">Nissim Maruani<sup>1,2</sup></a>&emsp;
<a href="https://s2.hk/">Bingchen Gong<sup>1</sup></a>&emsp;
<a href="https://www.lix.polytechnique.fr/~maks/">Maks Ovsjanikov<sup>1</sup></a>
</font>

<br>

<font size="3">
<sup>1</sup>Ecole Polytechnique, France &nbsp;&nbsp;
<sup>2</sup>Inria, Université Côte d'Azur, France
</font>

<br>

<font size="2">* Equal contribution</font>

<br>

[Project Page](https://diego1401.github.io/BlobsToSpokesWebsite/) | [arXiv](#) <!-- placeholder --> | [Download Meshes](https://drive.google.com/drive/folders/1Qc9qMzzPdXAUgggbSwVUMyXc_2A1DrYj?usp=sharing)

<br>

![Teaser](assets/teaser.png)

**Gaussian Wrapping** reconstructs watertight, textured surface meshes of full 3D scenes—including extremely thin structures such as bicycle spokes—at a fraction of the mesh size of concurrent works, by interpreting 3D Gaussians as stochastic oriented surface elements.


![Zoom Bike](assets/zoom_bike.gif)

</div>



## Installation

```bash
git clone --recurse-submodules https://github.com/diego1401/GaussianWrapping
cd GaussianWrapping
conda create -n gaussian_wrapping python=3.9 -y
conda activate gaussian_wrapping

# Set CUDA paths (adjust version as needed)
CUDA_VERSION=12.1 # or 11.8 (12.1 is best for 4090RTX)

export CPATH=/usr/local/cuda-${CUDA_VERSION}/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cuda-${CUDA_VERSION}/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-${CUDA_VERSION}/bin:$PATH

python install.py --cuda_version ${CUDA_VERSION}
```

# Training & Mesh Extraction

Gaussian Wrapping expects a **COLMAP-formatted dataset** (images + sparse reconstruction).
Use `-s` for the dataset root and `-m` for the output directory.

We provide two end-to-end scripts that run training followed by mesh extraction:

```bash
# Our median-depth rasterizer (faster, better metrics)
python gaussian_wrapping/scripts/train_and_extract_gw_ours.py \
    -s <PATH_TO_COLMAP_DATASET> \
    -m <OUTPUT_DIR> \
    -r 2

# RaDeGS rasterizer (smoother-looking meshes)
python gaussian_wrapping/scripts/train_and_extract_gw_radegs.py \
    -s <PATH_TO_COLMAP_DATASET> \
    -m <OUTPUT_DIR> \
    -r 2
```

Each script runs three steps in sequence: training, mesh extraction, and texture refinement. The refined mesh is saved alongside the extracted mesh with a `_texture_refined` suffix.

### Key options

| Flag | Description |
|------|-------------|
| `-r 2` | Downsample input images by 2× (used for metrics and comparisons) |

# Primal Adaptive Meshing

An alternative mesh extraction method that samples candidate points from an existing mesh (e.g. extracted by the standard pipeline), refines them onto the Gaussian occupancy isosurface via gradient descent, and reconstructs the surface with a Delaunay-based approach.

```bash
python gaussian_wrapping/primal_adaptive_meshing_extraction.py \
    -s <PATH_TO_COLMAP_DATASET> \
    -m <OUTPUT_DIR> \
    --input_mesh <PATH_TO_INPUT_MESH> \
    --output_mesh <PATH_TO_OUTPUT_MESH.ply> \
    --max_points <NUMBER_OF_MESH_VERTICES>
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input_mesh` | required | Input mesh to sample candidate points from (`.ply`) |
| `--output_mesh` | required | Output mesh path (must end in `.ply`) |
| `--max_points` | `1e6` | Number of candidate points to sample from the input mesh |
| `--bounding_box_method` | `scene` | `scene` uses the camera extent; `ground_truth` uses the TNT ground truth volume (if available); `blender` uses a custom volume exported from the Blender add-on (see below) |
| `--bounding_box_scaling` | `1.0` | Scale factor applied to the scene bounding box |
| `--bounding_box_file` | — | Path to the `.json` bounding volume exported from Blender (required when `--bounding_box_method blender`) |

<details>
<summary><strong>Advanced options</strong></summary>

| Flag | Default | Description |
|------|---------|-------------|
| `--n_steps` | `10` | Number of refinement steps |
| `--vacancy_threshold` | `0.1` | Points with vacancy further than this from 0.5 are filtered out |
| `--mesh_sampling_method` | `proportional_to_camera` | `proportional_to_camera` weights sampling by proximity to cameras; `surface_even` samples uniformly on the cropped section of the input mesh.|
| `--p_per_tet` | `10` | Points sampled per tetrahedron for occupancy estimation |
| `--oversampling_factor` | `2` | Mostly useful when `max_points` is small. Sample this multiple of `max_points` upfront per pass to reduce the number of expensive occupancy evaluations |
| `--post_process` | off | Apply post-processing to the output mesh |
| `--save_candidate_points` | off | Save the refined candidate point cloud alongside the mesh |
| `--plot_vacancy_histogram` | off | Save per-step vacancy histograms to the output directory |

</details>

## Blender Bounding Volume Add-on

If you want to extract an object of your scene at abritary resolution easily we propose this blender-add on to easily define the section of the scene to extract.


#### Installation
<details>

1. Open Blender.
2. Go to **Edit → Preferences → Add-ons → Install…**
3. Select `gaussian_wrapping/blender/bounding_volume_addon.py`.
4. Enable the **GW Bounding Volume Exporter** add-on in the list.

The add-on is now accessible as the **GW Bounds** tab in the 3D Viewport's N-panel (press **N** to open it).
</details>

#### Usage

1. **Import your input mesh** into Blender as a visual reference (e.g. *File → Import → Stanford PLY* and select the mesh you plan to pass to `--input_mesh`).

   ![Step 1](assets/step_1.png)

2. **Create a bounding mesh** — add any mesh (e.g. a Cube via *Shift+A → Mesh → Cube*) and transform (e.g edit mode → move vertices of cube) it so it encloses exactly the region of interest.

   ![Step 2](assets/step_2.png)
3. Open the **GW Bounds** tab in the N-panel:
   - Set **Bounding Mesh** to the object you just created.
   - Set **Export Path** to the `.json` file you want to write.
   - Click **Export Bounding Volume**.
4. Run the extraction script with:

```bash
python gaussian_wrapping/primal_adaptive_meshing_extraction.py \
    -s <PATH_TO_COLMAP_DATASET> \
    -m <OUTPUT_DIR> \
    --input_mesh <PATH_TO_INPUT_MESH> \
    --output_mesh <PATH_TO_OUTPUT_MESH.ply> \
    --bounding_box_method blender \
    --bounding_box_file <PATH_TO_EXPORTED.json>
```

   ![Step 3](assets/step_3.png)

5. **Decimate the output mesh** — the extracted mesh tends to be very dense. We recommend decimating it to reduce its size with no perceptible quality loss. This can be done via the script:

```bash
blender --background --python gaussian_wrapping/mesh_decimate.py -- \
    --in <PATH_TO_OUTPUT_MESH.ply> \
    --out <PATH_TO_DECIMATED_MESH.ply> \
    --ratio 0.3
```

   Alternatively, you can decimate directly in Blender: import the mesh (*File → Import → Stanford PLY*), select it, go to the **Properties** panel → **Modifier** tab, add a **Decimate** modifier, set the **Ratio** (e.g. `0.3`), and click **Apply**. Then export via *File → Export → Stanford PLY*.


## Benchmarks

### Tanks and Temples
<details>

> **Note on `_post_` meshes:** The meshes that are evaluated with the following metrics are passed through a post-processing procedure similar to [GGGS](https://github.com/HKUST-SAIL/Geometry-Grounded-Gaussian-Splatting) and [PGSR](https://zju3dv.github.io/pgsr/). This procedure yields in practice better metrics. However, we find that it can sometimes remove objects on the scene (e.g chandelier in the Meeting Room scene). Thus, for the textured meshes we recommend using non post-processed meshes.

To reproduce our full Tanks and Temples results, run the end-to-end benchmark scripts from the project root. Each script trains all 6 scenes, extracts meshes, and runs all three evaluations (uniform sampling, virtual scan sampling, and legacy TNT).

```bash
# Ours rasterizer
python gaussian_wrapping/scripts/benchmark_tnt_gw_ours.py \
    --data_dir <PATH_TO_TNT_SCENES> \
    --gt_dir <PATH_TO_TNT_GT> \
    --output_dir <OUTPUT_DIR>

# RaDeGS rasterizer
python gaussian_wrapping/scripts/benchmark_tnt_gw_radegs.py \
    --data_dir <PATH_TO_TNT_SCENES> \
    --gt_dir <PATH_TO_TNT_GT> \
    --output_dir <OUTPUT_DIR>
```

`<PATH_TO_TNT_SCENES>` should contain one subdirectory per scene (`Barn/`, `Caterpillar/`, etc.) in COLMAP format. `<PATH_TO_TNT_GT>` should contain the ground truth files for each scene (`<Scene>.ply`, `<Scene>.json`, `<Scene>_trans.txt`, `<Scene>_COLMAP_SfM.log`).

| Flag | Default | Description |
|------|---------|-------------|
| `--gpu_device` | `0` | CUDA device index |
| `--data_on_gpu` | off | Load dataset on GPU instead of CPU |
| `--depth_order` | off | Enable depth-order regularization (see note below) |
| `--depth_order_config` | — | Config name under `configs/depth_order/` (e.g. `default`, `strong`); only used with `--depth_order` |

> **Note on `--depth_order`:** This flag enables depth-order regularization using a pre-trained monocular depth model. It is **not used in the paper**, but can yield better results. It requires monocular depth priors to be pre-computed.

Per-scene results are written to `<OUTPUT_DIR>/<Scene>/`, with evaluation outputs in `eval_uniform/`, `eval_virtual_scan/`, and `eval_legacy/` subdirectories.

<details>
<summary><strong>Uniform and Virtual Scan Sampling Evaluations</strong></summary>

We provide evaluation scripts using two sampling strategies: **uniform sampling** and **virtual scan sampling**. Full documentation and the standalone toolbox are available at [diego1401/TNTUniScanEvals](https://github.com/diego1401/TNTUniScanEvals.git).

The scripts live under `gaussian_wrapping/eval/TNTUniScanEvals/`. See that directory's [README](gaussian_wrapping/eval/TNTUniScanEvals/README.md) for dataset setup and installation instructions.

#### Uniform Sampling Evaluation

Evaluates a mesh or point cloud against the ground truth using surface-based sampling and a 3-step ICP alignment pipeline.

```bash
python gaussian_wrapping/eval/TNTUniScanEvals/uniform_sampling_eval.py \
    --dataset-dir <path/to/scene_dir> \
    --traj-path <path/to/scene_dir>/<Scene>_COLMAP_SfM.log \
    --ply-path <path/to/mesh.ply> \
    --out-dir <output_dir>
```

#### Virtual Scan Sampling Evaluation

Evaluates a mesh by rendering depth from each training camera and projecting to world-space point clouds. Requires a COLMAP dataset and a CUDA-enabled GPU.

```bash
python gaussian_wrapping/eval/TNTUniScanEvals/virtual_scan_sampling_eval.py \
    -s <path/to/colmap_dataset> \
    -r 2 \
    --dataset-dir <path/to/scene_dir> \
    --traj-path <path/to/scene_dir>/<Scene>_COLMAP_SfM.log \
    --ply-path <path/to/mesh.ply> \
    --out-dir <output_dir>
```

Both scripts output **precision**, **recall**, and **F-score** to the console, along with precision/recall curve plots. Pass `--save_point_clouds` to also write color-coded error clouds.

</details>
</details>

### DTU

<details>
Coming soon.
</details>

### MipNeRF360

<details>
To reproduce our full MipNeRF 360 results, run the end-to-end benchmark scripts from the project root. Each script trains all 7 scenes, extracts meshes, and evaluates novel view synthesis quality.

```bash
# Ours rasterizer
python gaussian_wrapping/scripts/benchmark_mip360_gw_ours.py \
    --data_dir <PATH_TO_MIP360_SCENES> \
    --output_dir <OUTPUT_DIR>

# RaDeGS rasterizer
python gaussian_wrapping/scripts/benchmark_mip360_gw_radegs.py \
    --data_dir <PATH_TO_MIP360_SCENES> \
    --output_dir <OUTPUT_DIR>
```

`<PATH_TO_MIP360_SCENES>` should contain one subdirectory per scene (`bicycle/`, `bonsai/`, `counter/`, `garden/`, `kitchen/`, `room/`, `stump/`) in COLMAP format.

| Flag | Default | Description |
|------|---------|-------------|
| `--gpu_device` | `0` | CUDA device index |
| `--data_on_gpu` | off | Load dataset on GPU instead of CPU |
| `--depth_order` | off | Enable depth-order regularization (see note below) |
| `--depth_order_config` | — | Config name under `configs/depth_order/` (e.g. `default`, `strong`); only used with `--depth_order` |

> **Note on `--depth_order`:** This flag enables depth-order regularization using a pre-trained monocular depth model. It is **not used in the paper**, but can yield better results. It requires monocular depth priors to be pre-computed.

Per-scene results are written to `<OUTPUT_DIR>/<scene>/`. Rendered images and metrics (PSNR / SSIM / LPIPS) are saved in the standard Gaussian Splatting output structure.
</details>

### Mesh-Based Novel View Synthesis

<details>
We evaluate mesh-based novel view synthesis on the MipNeRF360 and Tanks and Temples datasets. Use the training and mesh extraction scripts described above to produce meshes for these datasets, then refer to [MILo](https://github.com/Anttwo/MILo.git) for the evaluation pipeline.

</details>

## Citation

```bibtex
@article{gomez2026gaussianwrapping,
  title={From Blobs to Spokes: High-Fidelity Surface Reconstruction via Oriented Gaussians},
  author={Gomez, Diego and Gu{\'e}don, Antoine and Maruani, Nissim and Gong, Bingchen and Ovsjanikov, Maks},
  journal={arXiv preprint arXiv:TODO_ARXIV_ID},
  year={2026}
}
```
