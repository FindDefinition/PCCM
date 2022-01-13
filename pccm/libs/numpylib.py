# Copyright 2022 Yan Yan
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import numpy as np
from pccm import FunctionCode
from pccm.builder.inliner import InlineBuilder, InlineBuilderPlugin
from pccm.utils import get_qualname_of_type
import numpy.typing as npt

class ArrayNumpy:
    def __init__(self, arr: np.ndarray, dtype: Optional[npt.DTypeLike] = None) -> None:
        assert arr.reshape(-1).shape[0] < 50, "array too large, must < 50"
        self.arr = arr
        self.dtype = np.dtype(dtype)
        

_NPDTYPE_TO_LIST_TYPE_STR: Dict[np.dtype, str] = {
    np.dtype(np.float16): "float",
    np.dtype(np.float32): "float",
    np.dtype(np.float64): "float",
    np.dtype(np.int8): "int64_t",
    np.dtype(np.int16): "int64_t",
    np.dtype(np.int32): "int64_t",
    np.dtype(np.int64): "int64_t",
    np.dtype(np.uint8): "int64_t",
    np.dtype(np.uint16): "int64_t",
    np.dtype(np.uint32): "int64_t",
    np.dtype(np.uint64): "int64_t",
    np.dtype(np.bool_): "bool",
}

class ArrayNumpyPlugin(InlineBuilderPlugin):
    """used to capture numpy to std::array.
    """
    QualifiedName = get_qualname_of_type(ArrayNumpy)
    def handle_captured_type(self, name: str, code: FunctionCode, obj: Any,
                                   user_arg: Optional[Any] = None) -> Optional[str]:
        return

    def type_conversion(self, obj: ArrayNumpy,
                                   user_arg: Optional[Any] = None):
        return obj.arr.tolist()

    def get_cpp_type(self, obj: ArrayNumpy,
                                   user_arg: Optional[Any] = None) -> str:
        ndim = obj.arr.ndim
        dtype = obj.dtype 
        if dtype is None:
            dtype = obj.arr.dtype 
        cpp_type = _NPDTYPE_TO_LIST_TYPE_STR[dtype]
        array_type = ""
        array_type += "std::array<" * ndim
        array_type += f"{cpp_type}, "
        shape_rev = obj.arr.shape[::-1]
        array_type += ">, ".join(map(str, shape_rev))
        # std::array<std::array<std::array<float, 3>, 4>, 2>
        return array_type + ">"

def main():
    import numpy as np 
    aa = np.zeros([2, 3, 4], dtype=np.float32)
    aa_array = ArrayNumpy(aa)
    b = InlineBuilder([], {ArrayNumpyPlugin.QualifiedName: ArrayNumpyPlugin()})
    for i in range(10):
        b.inline("just_a_name", f"""
        std::cout << $aa_array[0][0][0] << std::endl;
        """)

if __name__ == "__main__":
    main()
