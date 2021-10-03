import enum

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
                        mw_metas=[pccm.pybind.Pybind11PropMeta(name="hahaha")])
        self.add_member("private_prop_",
                        "int",
                        "0",
                        doc="a private prop. \ndetails.")

    @pccm.pybind.mark
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

    @pccm.pybind.mark_prop_getter(prop_name="square_prop")
    @pccm.member_function
    def prop_getter(self):
        code = pccm.FunctionCode("return private_prop_ * private_prop_;")
        return code.ret("int")

    @pccm.pybind.mark_prop_setter(prop_name="square_prop")
    @pccm.member_function
    def prop_setter(self):
        code = pccm.FunctionCode("private_prop_ = val;")
        code.arg("val", "int")
        return code


class Test4(Test3):
    def __init__(self):
        super().__init__()

    @pccm.pybind.mark(nogil=True)
    @pccm.member_function(inline=True)
    def add2(self):
        code = pccm.FunctionCode("")
        code.raw("""
        return add(a, b);
        """).arg("a,b", "int").ret("int")
        return code

    @pccm.pybind.mark
    @pccm.static_function
    def add_static(self):
        code = pccm.FunctionCode("")
        code.raw("""
        return a + b;
        """).arg("a,b", "int").ret("int")
        return code

    @pccm.pybind.mark(nogil=True)
    @pccm.member_function(inline=True, name="add2", const=True)
    def add2_overload(self):
        code = pccm.FunctionCode("")
        code.raw("""
        return a + b;
        """).arg("a,b", "int").ret("int")
        return code


class PbTestVirtual(pccm.Class, pccm.pybind.PybindClassMixin):
    def __init__(self):
        super().__init__()
        self.add_pybind_member("a", "int", "0")
        self.add_enum_class("EnumClassExample", [("kValue1", 1),
                                                 ("kValue2", 2)])
        self.add_enum("EnumExample", [("kValue1", 1), ("kValue2", 2)])
        self.add_member("bbb", "EnumClassExample", "EnumClassExample::kValue1")

    @pccm.pybind.mark(virtual=True)
    @pccm.member_function(virtual=True)
    def func_0(self):
        return pccm.FunctionCode("return 0;").ret("int")

    @pccm.pybind.mark(virtual=True)
    @pccm.member_function(virtual=True, pure_virtual=True)
    def func_2(self):
        code = pccm.FunctionCode("").ret("int")
        code.arg("a, b", "int")
        return code

    @pccm.pybind.mark
    @pccm.member_function
    def run_virtual_func_0(self):
        code = pccm.FunctionCode("return func_0();").ret("int")
        return code

    @pccm.pybind.mark
    @pccm.member_function
    def run_pure_virtual_func_2(self):
        code = pccm.FunctionCode("return func_2(a, b);").ret("int")
        code.arg("a,b", "int")
        return code
