from pccm import core
from pccm.libs import stl


class Test1(core.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(stl.STL)

    @core.member_function
    def add(self):
        return core.FunctionCode(
            """
        if (a > 0){
            return b;
        }
        return a + b;
        """, [core.Argument("a", "int"),
              core.Argument("b", "int")], "int")


class PTest1(core.ParameterizedClass):
    def __init__(self, const=5, type="float"):
        super().__init__()
        self.const = const
        self.type = type

    @core.member_function
    def add_template(self):
        argus = [core.Argument("a", self.type), core.Argument("b", self.type)]
        return core.FunctionCode("""
        return a + b + {};
        """.format(self.const),
                                 argus,
                                 return_type=self.type)


class Test2(core.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(Test1)

    @core.member_function
    def add(self):
        return core.FunctionCode(
            """
        auto tst1 = Test1();
        return tst1.add(a, b);
        """, [core.Argument("a", "int"),
              core.Argument("b", "int")], "int")
