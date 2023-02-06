import pccm

class BasicExample(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_member("prop_", "int", "0")

    @pccm.constructor
    def ctor(self):
        code = pccm.code()
        code.arg("p", "int")
        code.ctor_init("prop_", "p")
        code.raw(f"""
        prop_ += 3;
        """)
        return code

    @pccm.member_function
    def add(self):
        code = pccm.code()
        code.arg("a, b", "int")
        code.raw(f"""
        return a + b + prop_;
        """)
        return code.ret("int")

    @pccm.static_function
    def add_static(self):
        code = pccm.code()
        code.arg("a, b", "int")
        code.raw(f"""
        return a + b;
        """)
        return code.ret("int")


class BasicExampleParam(pccm.ParameterizedClass):
    def __init__(self, const=5, type="float"):
        super().__init__()
        self.const = const
        self.type = type

    @pccm.member_function
    def add_template(self):
        code = pccm.code()
        code.arg("a, b", self.type)
        code.raw(f"""
        return a + b + {self.const};
        """)
        return code.ret("int")

    @pccm.member_function
    def add_func_template(self):
        code = pccm.code()
        code.targ("T")
        code.arg("a,b", "T")
        code.raw(f"""
        return a + b + {self.const};
        """)
        code.ret("T")
        return code

# pccm.pybind.PybindClassMixin: add add_pybind_member 
class BasicExampleWithDepPybind(pccm.Class, pccm.pybind.PybindClassMixin):
    def __init__(self):
        super().__init__()
        self.add_dependency(BasicExample)
        self.add_param_class("ns1", BasicExampleParam(3, "int"), "PP")

        self.add_pybind_member("a", "int", "0")
        self.add_enum_class("EnumClassExample", [("kValue1", 1),
                                                 ("kValue2", 2)])
        self.add_enum("EnumExample", [("kValue1", 1), ("kValue2", 2)])
        self.add_member("bbb", "EnumClassExample", "EnumClassExample::kValue1")
        self.add_member("prop_", "int", "12")

    @pccm.pybind.mark
    @pccm.constructor
    def ctor(self):
        code = pccm.code()
        return code 

    @pccm.pybind.mark
    @pccm.member_function(inline=True)
    def add(self):
        code = pccm.code()
        code.arg("a,b", "int")
        with code.if_("a > 5"):
            code.raw("a += 3;")
        code.raw("""
        auto be = BasicExample(1);
        auto pp = PP();
        return be.add(a, b) + be.add_static(a, b) + pp.add_template(a, b) + pp.add_func_template(a, b);
        """)
        return code.ret("int")

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

