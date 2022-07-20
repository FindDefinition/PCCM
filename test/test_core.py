import shutil
from pathlib import Path

import ccimport
import pccm
from pccm import builder, core
from pccm.middlewares import pybind
from pccm.test_data.mod import PbTestVirtual, Test3, Test4, OtherSimpleClass
import pickle 
from ccimport import BuildMeta

def test_core():
    cu = Test4()
    cu2 = OtherSimpleClass()

    cu_scratch = core.Class()
    scratch_meta = core.StaticMemberFunctionMeta(name="scratch_func")
    scratch_code_obj = core.FunctionCode("")
    scratch_code_obj.raw("""
    return 50051;
    """).ret("int")
    cu_scratch.add_func_decl(core.FunctionDecl(scratch_meta, scratch_code_obj))
    cu_scratch.namespace = "scratch"
    cu_scratch.class_name = "ScratchClass"
    build_meta = BuildMeta()
    lib = builder.build_pybind(
        [cu_scratch, cu, cu2, PbTestVirtual()],
        Path(__file__).parent / "wtf2", build_meta)
    assert lib.pccm.test_data.mod.Test4.add_static(1, 2) == 3
    assert not hasattr(lib.pccm.test_data.mod.Test4, "invalid_method")
    t3 = lib.pccm.test_data.mod.Test3()
    t3.square_prop = 5
    assert t3.square_prop == 25

    class VirtualClass(lib.pccm.test_data.mod.PbTestVirtual):
        def func_0(self):
            self.a = 42
            return 0

        def func_2(self, a: int, b: int):
            self.a = a + b
            return 0

    vobj = VirtualClass()
    assert vobj.a == 0
    vobj.run_virtual_func_0()
    assert vobj.a == 42
    vobj.run_pure_virtual_func_2(3, 4)
    assert vobj.a == 7

    assert vobj.EnumExample.kValue1.name == "kValue1"
    assert vobj.EnumExample.kValue1.value == 1
    assert vobj.EnumExample.kValue1 | vobj.EnumExample.kValue2 == 3
    assert vobj.kValue1 | vobj.kValue2 == 3

    tsc = lib.pccm.test_data.mod.OtherSimpleClass()
    tsc.a = 444
    tsc_bytes = pickle.dumps(tsc)

    tsc_recover = pickle.loads(tsc_bytes)
    print(tsc_recover.a)


if __name__ == "__main__":
    test_core()
