import pccm
from pccm.libs import stl


class Test1(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(stl.STL)

    @pccm.member_function
    def add(self):
        return pccm.FunctionCode(
            """
        if (a > 0){
            return b;
        }
        return a + b;
        """, [pccm.Argument("a", "int"),
              pccm.Argument("b", "int")], "int")


class PTest1(pccm.ParameterizedClass):
    def __init__(self, const=5, type="float"):
        super().__init__()
        self.const = const
        self.type = type

    @pccm.member_function
    def add_template(self):
        argus = [pccm.Argument("a", self.type), pccm.Argument("b", self.type)]
        return pccm.FunctionCode("""
        return a + b + {};
        """.format(self.const),
                                 argus,
                                 return_type=self.type)


class Test2(pccm.Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(Test1)

    @pccm.member_function
    def add(self):
        return pccm.FunctionCode(
            """
        auto tst1 = Test1();
        return tst1.add(a, b);
        """, [pccm.Argument("a", "int"),
              pccm.Argument("b", "int")], "int")
