import enum

import pccm
from pccm.test_data import PTest1, Test2
from pccm.test_data.example import BasicExampleWithDepPybind


class Test3(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(Test2, BasicExampleWithDepPybind)
        self.add_param_class("ns1", PTest1(3, "int"), "PP")
        self.add_member("prop_",
                        "int",
                        "0",
                        doc="a private prop. \ndetails.")

    @pccm.pybind.mark
    @pccm.member_function(inline=True)
    def add(self):
        code = pccm.code("")
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
        code = pccm.code()
        code.raw(f"return prop_ * prop_;")
        return code.ret("int")

    @pccm.pybind.mark_prop_setter(prop_name="square_prop")
    @pccm.member_function
    def prop_setter(self):
        code = pccm.code()
        code.arg("val", "int")
        code.raw(f"prop_ = val;")
        return code

    @pccm.destructor
    def dtor(self):
        code = pccm.code("return;")
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

    @pccm.pybind.mark
    @pccm.static_function
    def invalid_method(self):
        code = pccm.FunctionCode("")
        code.raw("""
        return a + b;
        """).arg("a,b", "int").ret("int")
        return code.make_invalid()

class PbTestVirtual(pccm.Class, pccm.pybind.PybindClassMixin):
    def __init__(self):
        super().__init__()
        self.add_pybind_member("a", "int", "0")
        self.add_enum_class("EnumClassExample", [("kValue1", 1),
                                                 ("kValue2", 2)])
        self.add_enum("EnumExample", [("kValue1", 1), ("kValue2", 2)])
        self.add_member("bbb", "EnumClassExample", "EnumClassExample::kValue1")

        # self.build_meta.add_private_cflags("g++", "WTF")

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

class OtherSimpleClass(pccm.Class, pccm.pybind.PybindClassMixin):
    def __init__(self):
        super().__init__()
        self.add_pybind_member("a", "int", "0")
        self.add_raw_def(f"""
        pybind11::pickle(
            [](const {self.class_name} &p) {{ // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return pybind11::make_tuple(p.a);
            }},
            [](pybind11::tuple t) {{ // __setstate__
                if (t.size() != 1)
                    throw std::runtime_error("Invalid state!");

                /* Create a new C++ instance */
                {self.class_name} p{{t[0].cast<int>()}};
                return p;
            }}
        )
        """)