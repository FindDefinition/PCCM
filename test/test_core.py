import shutil
from pathlib import Path

import ccimport

from pccm import builder, core
from pccm.middlewares import pybind
from pccm.test_data.mod import Test3, Test4, PbTestVirtual


def test_core():
    cu = Test4()
    lib = builder.build_pybind([cu, PbTestVirtual()], Path(__file__).parent / "wtf2")
    assert lib.pccm.test_data.mod.Test4.add_static(1, 2) == 3
    t3 = lib.pccm.test_data.mod.Test3()
    t3.square_prop = 5
    assert t3.square_prop == 25


if __name__ == "__main__":
    test_core()
