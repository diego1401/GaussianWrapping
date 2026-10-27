"""Forward KB4 (OpenCV fisheye) camera of the rasterizer. Needs a CUDA device."""

import math

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from diff_gaussian_rasterization_ours import GaussianRasterizationSettings, GaussianRasterizer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

# Synthetic asymmetric intrinsics exercise the full image and rays beyond 90 degrees.
W, H = 720, 540
CAM = dict(fx=230.0, fy=225.0, cx=359.5, cy=269.5, k1=-0.05, k2=0.001, k3=-0.004, k4=0.0008)
THETA_MAX = math.radians(100.0)
KB4_THETA_MARGIN = 0.5  # config.h
VIEW_R = Rotation.from_euler("xyz", [20.0, -35.0, 50.0], degrees=True).as_matrix()
VIEW_T = np.array([0.3, -0.2, 0.5])
THETAS = (0.0, 5.0, 20.0, 40.0, 60.0, 75.0, 85.0, 89.9, 90.0, 90.1, 95.0, 98.0, 100.0)


def theta_d(theta):
    t2 = theta * theta
    return theta * (1 + t2 * (CAM["k1"] + t2 * (CAM["k2"] + t2 * (CAM["k3"] + t2 * CAM["k4"]))))


def project(p):
    """Pixels of camera-frame points (..., 3), in float64."""
    p = np.asarray(p, dtype=np.float64)
    rho = np.hypot(p[..., 0], p[..., 1])
    scale = np.divide(theta_d(np.arctan2(rho, p[..., 2])), rho, out=np.zeros_like(rho), where=rho > 0)
    return np.stack([CAM["fx"] * scale * p[..., 0] + CAM["cx"], CAM["fy"] * scale * p[..., 1] + CAM["cy"]], -1)


def jacobian(p, h=1e-6):
    """Full 2x3 Jacobian of project at p, by central differences."""
    return np.stack([(project(p + h * e) - project(p - h * e)) / (2 * h) for e in np.eye(3)], axis=-1)


def ray(u, v):
    """Unit ray of pixel (u, v); Newton's method inverts theta_d."""
    mx, my = (u - CAM["cx"]) / CAM["fx"], (v - CAM["cy"]) / CAM["fy"]
    td = math.hypot(mx, my)
    theta = td
    for _ in range(30):
        theta -= (theta_d(theta) - td) / ((theta_d(theta + 1e-7) - theta_d(theta - 1e-7)) / 2e-7)
    s = math.sin(theta) / td if td > 0 else 0.0
    return np.array([mx * s, my * s, math.cos(theta)])


def direction(theta_deg, phi_deg):
    t, p = math.radians(theta_deg), math.radians(phi_deg)
    return np.array([math.sin(t) * math.cos(p), math.sin(t) * math.sin(p), math.cos(t)])


def in_raster_directions(margin_px):
    for theta in THETAS:
        for phi in range(0, 360, 30) if theta else (0,):
            u, v = project(direction(theta, phi))
            if margin_px <= u <= W - 1 - margin_px and margin_px <= v <= H - 1 - margin_px:
                yield theta, phi


def render(p_cam, scales, rot_cam=None, opacity=0.3, **overrides):
    """Rasterize Gaussians given in camera coordinates; returns color, radii, depth, alpha, normal."""
    p_cam = np.atleast_2d(p_cam)
    n = len(p_cam)
    rot_cam = np.broadcast_to(np.eye(3), (n, 3, 3)) if rot_cam is None else np.asarray(rot_cam).reshape(n, 3, 3)
    w2c = np.eye(4)
    w2c[:3, :3], w2c[:3, 3] = VIEW_R, VIEW_T

    def cuda(a):
        return torch.tensor(np.asarray(a, dtype=np.float32), device="cuda")

    settings = dict(
        image_height=H, image_width=W, tanfovx=1.0, tanfovy=1.0, kernel_size=0.0, bg=cuda([0, 0, 0]),
        scale_modifier=1.0, viewmatrix=cuda(w2c.T), projmatrix=cuda(np.eye(4)), sh_degree=0, sg_degree=0,
        campos=cuda(-VIEW_R.T @ VIEW_T), prefiltered=False, require_depth=True, debug=False,
        camera_model="kb4", theta_max=THETA_MAX, **CAM,
    )
    settings.update(overrides)
    out = GaussianRasterizer(GaussianRasterizationSettings(**settings))(
        means3D=cuda((p_cam - VIEW_T) @ VIEW_R),
        means2D=torch.zeros((n, 3), device="cuda"),
        opacities=cuda(np.full((n, 1), opacity)),
        colors_precomp=cuda(np.ones((n, 3))),
        scales=cuda(np.broadcast_to(scales, (n, 3))),
        rotations=cuda(Rotation.from_matrix(VIEW_R.T @ rot_cam).as_quat()[:, [3, 0, 1, 2]]),
        sg_axis=torch.zeros((n, 0, 3), device="cuda"),
        sg_sharpness=torch.zeros((n, 0), device="cuda"),
        sg_color=torch.zeros((n, 0, 3), device="cuda"),
    )
    return [t.detach().cpu().numpy() for t in out]


def centroid(alpha):
    ys, xs = np.mgrid[: alpha.shape[1], : alpha.shape[2]]
    a = alpha[0].astype(np.float64)
    return np.array([(a * xs).sum(), (a * ys).sum()]) / a.sum()


def fit_splat(alpha, opacity):
    """Centre, 2x2 covariance and peak of one unclamped splat, from log(alpha) being quadratic in the pixel."""
    v, u = np.nonzero(alpha[0])
    u0, v0 = u.mean(), v.mean()
    x, y = u - u0, v - v0
    a, b, c, d, e, f = np.linalg.lstsq(
        np.stack([x * x, x * y, y * y, x, y, np.ones_like(x)], -1), np.log(alpha[0][v, u].astype(np.float64)), rcond=None
    )[0]
    conic = -2.0 * np.array([[a, b / 2], [b / 2, c]])
    centre = np.linalg.solve(conic, [d, e])
    return centre + [u0, v0], np.linalg.inv(conic), math.exp(f + 0.5 * centre @ conic @ centre) / opacity


def footprint_scale(p):
    """Isotropic 3D std whose smallest image std is 2 px."""
    return 2.0 / np.linalg.svd(jacobian(p), compute_uv=False)[1]


def test_projection_model_is_opencv_fisheye():
    cv2 = pytest.importorskip("cv2")
    p = np.array([[0.2, -0.1, 1.0], [-0.7, 0.4, 0.5], [0.05, 0.9, 0.3]])
    k = np.array([[CAM["fx"], 0, CAM["cx"]], [0, CAM["fy"], CAM["cy"]], [0, 0, 1]])
    d = np.array([CAM["k1"], CAM["k2"], CAM["k3"], CAM["k4"]])
    expected, _ = cv2.fisheye.projectPoints(p[:, None], np.zeros(3), np.zeros(3), k, d)
    np.testing.assert_allclose(project(p), expected[:, 0], atol=1e-6)


def test_mean_reprojects_within_0p1_px_up_to_100_deg():
    errors = {}
    for theta, phi in in_raster_directions(margin_px=20):
        p = 1.5 * direction(theta, phi)
        alpha = render(p, footprint_scale(p))[3]
        errors[theta, phi] = np.linalg.norm(centroid(alpha) - project(p))
    assert {theta for theta, _ in errors} == set(THETAS)
    per_theta = {t: max(e for (theta, _), e in errors.items() if theta == t) for t in THETAS}
    print(f"\n{len(errors)} directions, worst centroid error [px] per theta [deg]:", per_theta)
    assert max(errors.values()) <= 0.1


@pytest.mark.parametrize("kernel_size", [0.0, 0.3])
def test_footprint_is_the_ewa_of_the_full_jacobian(kernel_size):
    rng = np.random.default_rng(0)
    for theta, phi in [(0.0, 0), (45.0, 120), (89.9, 0), (90.0, 0), (90.0, 210), (95.0, 330), (100.0, 30), (100.0, 150)]:
        p = 1.5 * direction(theta, phi)
        rot = Rotation.random(random_state=rng).as_matrix()
        scales = footprint_scale(p) * np.array([1.0, 1.7, 2.6])
        alpha = render(p, scales, rot, kernel_size=kernel_size)[3]
        centre, cov, peak = fit_splat(alpha, opacity=0.3)
        j = jacobian(p)
        ewa = j @ rot @ np.diag(scales**2) @ rot.T @ j.T
        expected = ewa + kernel_size * np.eye(2)
        assert np.linalg.norm(centre - project(p)) <= 1e-3, (theta, phi)
        assert np.linalg.norm(cov - expected) <= 1e-3 * np.linalg.norm(expected), (theta, phi, cov, expected)
        assert peak == pytest.approx(math.sqrt(np.linalg.det(ewa) / np.linalg.det(expected)), rel=1e-3)


def test_depth_is_the_distance_along_each_pixel_ray():
    for theta, phi in [(0.0, 0), (45.0, 300), (90.0, 0), (100.0, 30), (100.0, 150)]:
        p = 1.5 * direction(theta, phi)
        # A thin disk tilted 35 deg off the line of sight: its median depth is the ray-plane distance.
        z = -p / np.linalg.norm(p)
        axis = np.cross(z, [0.3, 0.9, 0.1])
        normal = Rotation.from_rotvec(np.radians(35.0) * axis / np.linalg.norm(axis)).apply(z)
        tangent = np.cross(normal, [0.0, 0.0, 1.0] if abs(normal[2]) < 0.9 else [1.0, 0.0, 0.0])
        tangent /= np.linalg.norm(tangent)
        rot = np.stack([tangent, np.cross(normal, tangent), normal], axis=-1)
        _, _, depth, alpha, rendered_normal = render(p, footprint_scale(p) * np.array([5.0, 5.0, 1e-5]), rot, opacity=0.99)

        def plane_distance(u, v):
            r = ray(u, v)
            return normal @ p / (normal @ r)

        u0, v0 = project(p)
        grad = np.array([plane_distance(u0 + 1e-3, v0) - plane_distance(u0 - 1e-3, v0),
                         plane_distance(u0, v0 + 1e-3) - plane_distance(u0, v0 - 1e-3)]) / 2e-3
        cu, cv = int(round(u0)), int(round(v0))
        for du, dv in [(0, 0), (3, 0), (-3, 0), (0, 3), (0, -3)]:
            u, v = cu + du, cv + dv
            assert alpha[0, v, u] > 0.6
            linear = plane_distance(u0, v0) + grad @ [u - u0, v - v0]
            assert depth[0, v, u] == pytest.approx(linear, abs=2e-4), (theta, phi, du, dv)
            assert depth[0, v, u] == pytest.approx(plane_distance(u, v), rel=2e-3), (theta, phi, du, dv)
        assert rendered_normal[:, cv, cu] @ normal == pytest.approx(1.0, abs=1e-5), (theta, phi)


def test_culls_near_means_and_means_beyond_the_margin():
    edge = math.degrees(THETA_MAX + KB4_THETA_MARGIN)
    kept, culled = 1.5 * direction(edge - 2.0, 36), 1.5 * direction(edge + 1.5, 36)
    assert project(culled)[0] < W - 20 and project(culled)[1] < H - 20
    radii = render(np.stack([kept, culled, [0.0, 0.0, 0.15], [0.0, 0.0, 0.25]]), np.full(3, 0.01))[1]
    assert (radii > 0).tolist() == [True, False, False, True]


def test_matches_the_pinhole_camera_near_the_axis():
    # With k1..k4 = 0, kb4 maps theta to f * theta and a pinhole to f * tan(theta): 0.012 px apart at 3 deg.
    f = CAM["fx"]
    znear, zfar = 0.01, 100.0
    proj = np.zeros((4, 4))
    proj[0, 0], proj[1, 1] = 2 * f / W, 2 * f / H
    proj[2, 2], proj[2, 3], proj[3, 2] = zfar / (zfar - znear), -zfar * znear / (zfar - znear), 1.0
    w2c = np.eye(4)
    w2c[:3, :3], w2c[:3, 3] = VIEW_R, VIEW_T
    pinhole_camera = dict(camera_model="pinhole", tanfovx=W / (2 * f), tanfovy=H / (2 * f),
                          projmatrix=torch.tensor((proj @ w2c).T, dtype=torch.float32, device="cuda"))
    kb4_camera = dict(fx=f, fy=f, cx=(W - 1) / 2, cy=(H - 1) / 2, k1=0.0, k2=0.0, k3=0.0, k4=0.0)
    ys, xs = np.mgrid[:H, :W]
    rln = 1 / np.sqrt(((xs - (W - 1) / 2) / f) ** 2 + ((ys - (H - 1) / 2) / f) ** 2 + 1)
    rng = np.random.default_rng(1)
    for _ in range(8):
        p = rng.uniform(1.0, 2.0) * direction(rng.uniform(0.0, 3.0), rng.uniform(0.0, 360.0))
        rot, scales = Rotation.random(random_state=rng).as_matrix(), rng.uniform(0.01, 0.03, 3)
        pinhole = render(p, scales, rot, opacity=0.99, **pinhole_camera)
        kb4 = render(p, scales, rot, opacity=0.99, **kb4_camera)
        covered = (pinhole[3][0] > 0.6) & (kb4[3][0] > 0.6)
        assert covered.sum() >= 10
        np.testing.assert_allclose(kb4[0], pinhole[0], atol=1e-2)
        np.testing.assert_allclose(kb4[3], pinhole[3], atol=1e-2)
        np.testing.assert_allclose(kb4[4][:, covered], pinhole[4][:, covered], atol=1e-2)
        np.testing.assert_allclose(kb4[2][0][covered] * rln[covered], pinhole[2][0][covered], rtol=1e-3)


def test_rejects_unknown_models_bad_intrinsics_and_gradients():
    p = np.array([[0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="camera_model"):
        render(p, np.full(3, 0.01), camera_model="fisheye")
    with pytest.raises(RuntimeError, match="theta_max"):
        render(p, np.full(3, 0.01), theta_max=3.0)
    means = torch.tensor([[0.0, 0.0, 1.0]], device="cuda", requires_grad=True)
    rasterizer = GaussianRasterizer(GaussianRasterizationSettings(
        image_height=H, image_width=W, tanfovx=1.0, tanfovy=1.0, kernel_size=0.0,
        bg=torch.zeros(3, device="cuda"), scale_modifier=1.0, viewmatrix=torch.eye(4, device="cuda"),
        projmatrix=torch.eye(4, device="cuda"), sh_degree=0, sg_degree=0, campos=torch.zeros(3, device="cuda"),
        prefiltered=False, require_depth=True, debug=False, camera_model="kb4", theta_max=THETA_MAX, **CAM))
    color = rasterizer(
        means3D=means, means2D=torch.zeros((1, 3), device="cuda"), opacities=torch.full((1, 1), 0.5, device="cuda"),
        colors_precomp=torch.ones((1, 3), device="cuda"), scales=torch.full((1, 3), 0.01, device="cuda"),
        rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda"), sg_axis=torch.zeros((1, 0, 3), device="cuda"),
        sg_sharpness=torch.zeros((1, 0), device="cuda"), sg_color=torch.zeros((1, 0, 3), device="cuda"))[0]
    with pytest.raises(NotImplementedError, match="backward"):
        color.sum().backward()
    with pytest.raises(NotImplementedError, match="markVisible"):
        rasterizer.markVisible(means.detach())
    for operation in ("integrate", "evaluate_sdf", "sample_depth"):
        with pytest.raises(NotImplementedError, match=operation):
            getattr(rasterizer, operation)(means.detach(), means.detach(), torch.ones((1, 1), device="cuda"))
