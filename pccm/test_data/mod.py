from pccm import core
from pccm.middlewares import pybind
from pccm.test_data import PTest1, Test2


class Test3(core.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(Test2)
        self.add_param_class("ns1", PTest1(3, "int"), "PP")

    @pybind.pybind_mark
    @core.member_function(inline=True)
    def add(self):
        code = core.FunctionCode("")
        with code.if_("a > 5"):
            code.raw("a += 3;")

        code.raw("""
        auto tst1 = Test2();
        auto tst2 = PP();
        return tst1.add(a, b) + tst2.add_template(a, b);
        """).arg("a,b", "int").ret("int")
        return code


class Test4(Test3):
    def __init__(self):
        super().__init__()

    @pybind.pybind_mark
    @core.member_function(inline=True)
    def add2(self):
        code = core.FunctionCode("")
        code.raw("""
        return add(a, b);
        """).arg("a,b", "int").ret("int")
        return code
