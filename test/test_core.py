from pccm import core 
from pccm.test_data.mod import Test3
from pathlib import Path 
from pccm.middlewares import pybind
from pccm import builder
import shutil 
import ccimport
def test_core():
    pb = pybind.Pybind11("wtf", "wtf")
    cg = core.CodeGenerator([pb])
    cu = Test3()
    cg.build_graph([cu])
    header_dict, impl_dict = cg.code_generation(cg.get_code_units())
    HEADER_ROOT = Path(__file__).parent / "build" / "include"
    SRC_ROOT = Path(__file__).parent / "build" / "src"
    cg.code_written(HEADER_ROOT, header_dict)
    paths = cg.code_written(SRC_ROOT, impl_dict)
    cus = pb.get_code_units()
    for cu in cus:
        print(cu.namespace)
    header_dict, impl_dict = cg.code_generation(cus)
    cg.code_written(HEADER_ROOT, header_dict)
    paths += cg.code_written(SRC_ROOT, impl_dict)

    lib = ccimport.ccimport(paths, Path(__file__).parent / "build" / "wtf", 
        [HEADER_ROOT])
    

def test_builder():
    cu = Test3()
    lib = builder.build_pybind([cu], Path(__file__).parent / "build" / "wtf")


if __name__ == "__main__":
    test_core()