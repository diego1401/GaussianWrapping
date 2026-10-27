# Forward KB4 camera

- `GaussianRasterizationSettings` accepts `camera_model="kb4"`, `fx`, `fy`, `cx`, `cy`, `k1` through `k4`, and `theta_max` in radians. Pixel centers use integer coordinates.
- `render_ours(..., camera_settings={...})` forwards these fields to the rasterizer.
- KB4 returns median depth as distance along the pixel ray in scene units. Pinhole depth retains its optical-axis convention.
- KB4 is forward-only. Backward, `markVisible`, `integrate`, `evaluate_sdf`, and `sample_depth` raise `NotImplementedError`.
- Means closer than 0.2 scene units are culled. Means beyond `theta_max + 0.5` radians are culled; the caller must mask pixels outside `theta_max`.
- The pinhole camera remains the default.

## Build and test

Install the upstream runtime dependencies, then rebuild this extension with a CUDA toolkit compatible with the installed PyTorch:

```bash
python -m pip install --no-build-isolation ./submodules/diff-gaussian-rasterization_ours
python -m pytest -q -s submodules/diff-gaussian-rasterization_ours/tests/test_kb4.py
```

- The tests require CUDA. They skip without a CUDA device.
- The camera and geometry fixtures are synthetic. Coverage includes projection through 100 degrees, full-Jacobian footprints, radial depth, culling, pinhole agreement, and unsupported operations.
- The implementation was checked with Python 3.11, PyTorch 2.7.1/CUDA 12.8, and an RTX 5070 Ti. The upstream installer’s CUDA 11.8/12.1 combinations were not tested for this extension.

## Pinhole byte comparison

Prepare separate environments with the base and KB4 extensions. Run the same script and fixture in both environments:

```bash
python submodules/diff-gaussian-rasterization_ours/tests/synthetic_ply.py synthetic.ply
<base-python> submodules/diff-gaussian-rasterization_ours/tests/pinhole_regression.py render <base-checkout> synthetic.ply base1.npz
<base-python> submodules/diff-gaussian-rasterization_ours/tests/pinhole_regression.py render <base-checkout> synthetic.ply base2.npz
<kb4-python> submodules/diff-gaussian-rasterization_ours/tests/pinhole_regression.py render <kb4-checkout> synthetic.ply kb4.npz
python submodules/diff-gaussian-rasterization_ours/tests/pinhole_regression.py compare base1.npz base2.npz
python submodules/diff-gaussian-rasterization_ours/tests/pinhole_regression.py compare base1.npz kb4.npz
```

Each comparison checks 30 arrays: color, radii, depth, alpha, and normals at three camera views with depth enabled and disabled. A difference produces a nonzero exit status.
