"""Render fixed pinhole views of a 3DGS PLY through render_ours, and compare two runs byte for byte.

    python pinhole_regression.py render <GaussianWrapping checkout> <scene.ply> <out.npz>
    python pinhole_regression.py compare <a.npz> <b.npz>

Run `render` once per build of diff_gaussian_rasterization_ours, each with the checkout it was built from.
"""

import hashlib
import math
import sys
from pathlib import Path

import numpy as np

# width, height, fovy [deg], camera position as a fraction of the 5-95 % box of the means, forward, up
VIEWS = (
    (719, 719, 90.0, (0.5, 0.5, 0.5), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    (640, 480, 60.0, (0.15, 0.2, 0.7), (1.0, 1.0, -0.6), (0.0, 0.0, 1.0)),
    (1078, 808, 120.0, (0.5, 0.6, 0.4), (-0.3, -1.0, -0.2), (0.0, 0.0, 1.0)),
)
OUTPUTS = ("color", "radii", "depth", "alpha", "normal")


def render(gw_root, ply, out):
    gw = Path(gw_root).resolve()
    for path in (gw / "gaussian_wrapping", gw, gw / "submodules"):
        sys.path.insert(0, str(path))
    import torch
    import diff_gaussian_rasterization_ours as dgr
    from gaussian_renderer import GaussianModel
    from gaussian_renderer.ours import render_ours
    from utils.graphics_utils import getProjectionMatrix, getWorld2View2

    so = Path(dgr._C.__file__)
    print(f"extension {so} sha256 {hashlib.sha256(so.read_bytes()).hexdigest()[:16]}")
    captured = []
    torch.nn.modules.module.register_module_forward_hook(
        lambda module, args, output: captured.append(output) if isinstance(module, dgr.GaussianRasterizer) else None
    )

    class Pipe:
        convert_SHs_python = False
        compute_cov3D_python = False
        debug = False

    class Cam:
        pass

    gaussians = GaussianModel(3)
    gaussians.load_ply(str(ply))
    lo, hi = np.percentile(gaussians.get_xyz.detach().cpu().numpy(), [5, 95], axis=0)
    bg = torch.tensor([0.2, 0.4, 0.6], device="cuda")
    arrays = {}
    with torch.inference_mode():
        for i, (width, height, fovy_deg, frac, forward, up) in enumerate(VIEWS):
            # The GS server's camera: rows right, -up, forward map world to camera.
            pos = lo + (hi - lo) * np.array(frac)
            forward = np.array(forward) / np.linalg.norm(forward)
            right = np.cross(forward, up)
            right /= np.linalg.norm(right)
            r_w2c = np.stack([right, -np.cross(right, forward), forward])
            fovy = math.radians(fovy_deg)
            fovx = 2.0 * math.atan(math.tan(fovy / 2.0) * width / height)
            cam = Cam()
            cam.FoVx, cam.FoVy, cam.image_width, cam.image_height = fovx, fovy, width, height
            cam.world_view_transform = torch.tensor(getWorld2View2(r_w2c.T, -r_w2c @ pos, np.zeros(3), 1.0)).transpose(0, 1).cuda()
            proj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
            cam.full_proj_transform = cam.world_view_transform.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
            cam.camera_center = cam.world_view_transform.inverse()[3, :3]
            for require_depth in (True, False):
                captured.clear()
                render_ours(cam, gaussians, Pipe(), bg, require_depth=require_depth)
                for name, tensor in zip(OUTPUTS, captured[0]):
                    arrays[f"view{i}_depth{int(require_depth)}_{name}"] = tensor.cpu().numpy()
            alpha, radii = arrays[f"view{i}_depth1_alpha"], arrays[f"view{i}_depth1_radii"]
            print(f"view{i} {width}x{height} fovy {fovy_deg}: {int((radii > 0).sum())} Gaussians, mean alpha {alpha.mean():.3f}")
    np.savez(out, **arrays)


def compare(a, b):
    a, b = np.load(a), np.load(b)
    assert sorted(a.files) == sorted(b.files), (a.files, b.files)
    differing = []
    for key in sorted(a.files):
        x, y = a[key], b[key]
        if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
            differing.append(key)
            print(f"{key}: differs, max |a - b| = {np.nanmax(np.abs(x.astype(np.float64) - y)):.3g}")
    print(f"{len(a.files) - len(differing)}/{len(a.files)} arrays byte-identical")
    return 1 if differing else 0


if __name__ == "__main__":
    command, *args = sys.argv[1:]
    sys.exit(render(*args) if command == "render" else compare(*args))
