from setuptools import setup
import os
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME

ROOT = os.path.dirname(os.path.abspath(__file__))

# Build CUDA code only if torch supports CUDA and the CUDA toolkit (nvcc) is available.
# Set LIETORCH_FORCE_CPU=1 to force a CPU-only build.
FORCE_CPU = os.environ.get("LIETORCH_FORCE_CPU", "0") == "1"
WITH_CUDA = (not FORCE_CPU) and torch.version.cuda is not None and CUDA_HOME is not None
print(f"lietorch: building {'with' if WITH_CUDA else 'without'} CUDA support")

include_dirs = [
    os.path.join(ROOT, "lietorch/include"),
    os.path.join(ROOT, "eigen"),
]
sources = [
    "lietorch/src/lietorch.cpp",
    "lietorch/src/lietorch_cpu.cpp",
]

if WITH_CUDA:
    ext_modules = [
        CUDAExtension("lietorch_backends",
            include_dirs=include_dirs,
            sources=sources + ["lietorch/src/lietorch_gpu.cu"],
            extra_compile_args={
                "cxx": ["-O2", "-DWITH_CUDA"],
                "nvcc": ["-O2", "-DWITH_CUDA"],
            }),

        CUDAExtension("lietorch_extras",
            sources=[
                "lietorch/extras/altcorr_kernel.cu",
                "lietorch/extras/corr_index_kernel.cu",
                "lietorch/extras/se3_builder.cu",
                "lietorch/extras/se3_inplace_builder.cu",
                "lietorch/extras/se3_solver.cu",
                "lietorch/extras/extras.cpp",
            ],
            extra_compile_args={
                "cxx": ["-O2"],
                "nvcc": ["-O2"],
            }),
    ]
else:
    ext_modules = [
        CppExtension("lietorch_backends",
            include_dirs=include_dirs,
            sources=sources,
            extra_compile_args={"cxx": ["-O2"]}),
    ]

setup(
    name="lietorch",
    version="0.3",
    description="Lie Groups for PyTorch",
    author="Zachary Teed",
    packages=["lietorch"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
