import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="diff_gaussian_rasterization_ours",
    packages=["diff_gaussian_rasterization_ours"],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization_ours._C",
            sources=[
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/render_forward.cu",
                "cuda_rasterizer/render_backward.cu",
                "cuda_rasterizer/sample_forward.cu",
                "cuda_rasterizer/sample_backward.cu",
                "rasterize_points.cu",
                "ext.cpp",
            ],
            extra_compile_args={
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-std=c++17",
                    "--extended-lambda",
                    # "--Werror=all-warnings",
                    "--expt-relaxed-constexpr",
                    "-I" + os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "third_party/glm/",
                    ),
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)