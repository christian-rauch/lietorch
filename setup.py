from setuptools import setup
import os
import glob
import subprocess
import sys
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME, ROCM_HOME

ROOT = os.path.dirname(os.path.abspath(__file__))

# Select the GPU backend from the installed PyTorch build and toolkit:
#   ROCm: torch.version.hip is set and ROCM_HOME (hipcc) is available
#   CUDA: torch.version.cuda is set and CUDA_HOME (nvcc) is available
#   CPU:  otherwise
# Set LIETORCH_FORCE_CPU=1 to force a CPU-only build.
# On ROCm, torch's CUDAExtension hipifies the CUDA sources during the build.
FORCE_CPU = os.environ.get("LIETORCH_FORCE_CPU", "0") == "1"
if FORCE_CPU:
    BACKEND = "cpu"
elif getattr(torch.version, "hip", None) is not None and ROCM_HOME is not None:
    BACKEND = "rocm"
elif torch.version.cuda is not None and CUDA_HOME is not None:
    BACKEND = "cuda"
else:
    BACKEND = "cpu"
WITH_GPU = BACKEND in ("cuda", "rocm")
print(f"lietorch: building with {BACKEND.upper()} backend")

include_dirs = [
    os.path.join(ROOT, "lietorch/include"),
    os.path.join(ROOT, "eigen"),
]
sources = [
    "lietorch/src/lietorch.cpp",
    "lietorch/src/lietorch_cpu.cpp",
]

def rocm_build_paths():
    """Extra include/library dirs needed by pip-installed ROCm SDKs, which ship
    neither thrust headers (rocm-sdk-devel) nor an unversioned libamdhip64.so."""
    include_dirs, library_dirs = [], []
    roots = [ROCM_HOME]
    # rocm-sdk-devel unpacks itself lazily on the first `rocm-sdk path` call
    try:
        out = subprocess.run([sys.executable, "-m", "rocm_sdk", "path", "--root"],
                             capture_output=True, text=True)
        if out.returncode == 0 and out.stdout.strip():
            roots.append(out.stdout.strip().splitlines()[-1])
    except OSError:
        pass
    inc = [os.path.join(r, "include") for r in roots if r]
    inc = [d for d in inc if os.path.isfile(os.path.join(d, "thrust", "complex.h"))]
    if not inc:
        raise RuntimeError(
            "ROCm build requires the thrust headers (thrust/complex.h). Install the ROCm "
            "development files matching your PyTorch, e.g. `pip install rocm[devel]==<version>` "
            "from the same index as torch, or a system ROCm with rocthrust.")
    include_dirs.append(inc[0])

    lib_dir = os.path.join(ROCM_HOME, "lib") if ROCM_HOME else None
    if lib_dir and not os.path.exists(os.path.join(lib_dir, "libamdhip64.so")):
        versioned = sorted(glob.glob(os.path.join(lib_dir, "libamdhip64.so.*")))
        if versioned:
            link_dir = os.path.join(ROOT, "build", "rocm_links")
            os.makedirs(link_dir, exist_ok=True)
            link = os.path.join(link_dir, "libamdhip64.so")
            if os.path.lexists(link):
                os.remove(link)
            os.symlink(versioned[0], link)
            library_dirs.append(link_dir)
    return include_dirs, library_dirs


if WITH_GPU:
    rocm_includes, rocm_libs = rocm_build_paths() if BACKEND == "rocm" else ([], [])
    include_dirs = include_dirs + rocm_includes
    # PyTorch exposes ROCm devices as torch.device("cuda"), so WITH_CUDA enables
    # the GPU dispatch in both cases; WITH_ROCM additionally marks HIP builds.
    gpu_flags = ["-O2", "-DWITH_CUDA"] + (["-DWITH_ROCM"] if BACKEND == "rocm" else [])
    ext_modules = [
        CUDAExtension("lietorch_backends",
            include_dirs=include_dirs,
            library_dirs=rocm_libs,
            sources=sources + ["lietorch/src/lietorch_gpu.cu"],
            extra_compile_args={
                "cxx": gpu_flags,
                "nvcc": gpu_flags,  # compiled with hipcc on ROCm
            }),

        CUDAExtension("lietorch_extras",
            include_dirs=include_dirs,
            library_dirs=rocm_libs,
            sources=[
                "lietorch/extras/altcorr_kernel.cu",
                "lietorch/extras/corr_index_kernel.cu",
                "lietorch/extras/se3_builder.cu",
                "lietorch/extras/se3_inplace_builder.cu",
                "lietorch/extras/se3_solver.cu",
                "lietorch/extras/extras.cpp",
            ],
            extra_compile_args={
                "cxx": gpu_flags,
                "nvcc": gpu_flags,
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
