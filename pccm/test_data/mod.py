import pccm
from pccm.test_data import PTest1, Test2


class Test3(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(Test2)
        self.add_param_class("ns1", PTest1(3, "int"), "PP")
        self.add_member("hahaha",
                        "int",
                        pyanno="int",
                        mw_metas=[pccm.pybind.Pybind11PropMeta()])

    @pccm.pybind.pybind_mark
    @pccm.member_function(inline=True)
    def add(self):
        code = pccm.FunctionCode("")
        with code.if_("a > 5"):
            code.raw("a += 3;")

        code.raw("""
        auto tst1 = Test2();
        auto tst2 = PP();
        return tst1.add(a, b) + tst2.add_template(a, b) + tst2.add_func_template(a, b);
        """).arg("a,b", "int").ret("int")
        return code


class Test4(Test3):
    def __init__(self):
        super().__init__()
        self.set_this_class_type(__class__)

    @pccm.pybind.pybind_mark(nogil=True)
    @pccm.member_function(inline=True)
    def add2(self):
        code = pccm.FunctionCode("")
        code.raw("""
        return add(a, b);
        """).arg("a,b", "int").ret("int")
        return code

    @pccm.pybind.pybind_mark
    @pccm.static_function
    def add_static(self):
        code = pccm.FunctionCode("")
        code.raw("""
        return a + b;
        """).arg("a,b", "int").ret("int")
        return code
