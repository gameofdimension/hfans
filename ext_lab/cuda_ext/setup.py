from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='lltm_cuda',
    ext_modules=[
        CUDAExtension('lltm_cuda', [
            'cuda.cpp',
            'cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })