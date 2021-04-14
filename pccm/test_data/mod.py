from pccm.test_data import PTest1, Test2
from pccm import core
from pccm.middlewares import pybind 

class Test3(core.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(Test2)
        self.add_param_class("ns1", PTest1(3, "int"))

    @pybind.pybind_mark
    @core.member_function(inline=True)
    def add(self):
        return core.FunctionCode("""
        auto tst1 = Test2();
        auto tst2 = ns1::PTest1();
        return tst1.add(a, b) + tst2.add_template(a, b);
        """, [core.Argument("a", "int"), core.Argument("b", "int")],
        return_type="int")
