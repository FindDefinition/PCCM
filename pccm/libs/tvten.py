from typing import List, Optional, Set, Union
from pccm.core import FunctionCode, register_arg_attr_hook, FuncArgAttrHandler
from pccm.core.funccode import Argument

@register_arg_attr_hook("tv::Tensor")
class TVTenHandler(FuncArgAttrHandler):
    def __init__(self) -> None:
        super().__init__()
        self._shape_keys: Set[str] = set()

    def handle(self, code: "FunctionCode", args: List[Argument]):
        """tv::Tensor attrs:
        dtype, ndim, shape
        """
        shape_keys: Set[str] = self._shape_keys
        for arg in args:
            attrs = arg.attrs
            if not attrs:
                continue
            dtype = str(attrs[0].value)
            ndim: Optional[int] = None
            shape: Optional[List[Union[int, str]]] = None
            if len(attrs) >= 2:
                if isinstance(attrs[1].value, list):
                    shape = attrs[1].value
                    ndim = len(shape)
                else:
                    ndim = int(attrs[1].value)
            if len(attrs) >= 3:
                assert not isinstance(attrs[1].value, list)
                assert isinstance(attrs[2].value, list)
                shape = attrs[2].value
            if dtype != "void":
                code.raw(f"""
                {dtype}* {arg.name}_ptr = {arg.name}.data_ptr<{dtype}>();
                """)
            if ndim is not None:
                code.raw(f"""
                TV_ASSERT_INVALID_ARG({arg.name}.ndim() == {ndim}, 
                    "{arg.name} ndim", {arg.name}.ndim(), "not equal to {ndim}")
                """)
            if shape is not None:
                assert ndim == len(shape), f"ndim {ndim} shape {shape} mismatch"
                # 1. constant shape checkers
                checkers: List[str] = []
                for i, s in enumerate(shape):
                    if isinstance(s, int):
                        if s != -1:
                            checkers.append(f"{arg.name}.dim({i}) == {s}")
                    else:
                        # s is str, export it
                        if s in shape_keys:
                            code.raw(f"""
                            TV_ASSERT_INVALID_ARG({s} == {arg.name}.dim({i}), 
                                "{arg.name}.shape[{i}] mismatch. expected:", {s}, ", get:", {arg.name}.dim({i}));
                            """)
                        else:
                            shape_keys.add(s)
                            code.raw(f"auto {s} = {arg.name}.dim({i});")
                if checkers:
                    check_stmt = " && ".join(checkers)
                    shape_str = ", ".join(map(str, shape))
                    code.raw(f"""
                    TV_ASSERT_INVALID_ARG({check_stmt}, 
                        "shape mismatch. expected: [{shape_str}], get:", {arg.name}.shape());
                    """)



if __name__ == "__main__":
    code = FunctionCode()

    code.arg("a[\"const float\", 2, (K, 3)], b[int, 1, (K)]", "tv::Tensor")
    print(code.inspect_body())
    # print(code._)