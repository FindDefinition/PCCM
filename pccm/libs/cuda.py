from pccm.core import Class


class CUDADriver(Class):
    def __init__(self):
        self.add_include("cuda_runtime_api.h")
        self.add_include("cuda.h")


class CUDADevice(Class):
    def __init__(self):
        self.add_include("cuda_fp16.h")


class CUBLAS(Class):
    def __init__(self):
        self.add_include("cublas.h")
        self.add_include("cublaslt.h")
