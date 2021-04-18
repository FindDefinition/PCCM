from pccm.core import Class


class STLContainers(Class):
    def __init__(self):
        super().__init__()
        self.add_include("vector")
        self.add_include("array")
        self.add_include("map")
        self.add_include("set")
        self.add_include("unordered_map")
        self.add_include("unordered_set")


class STLMemory(Class):
    def __init__(self):
        super().__init__()
        self.add_include("memory")


class STLIO(Class):
    def __init__(self):
        super().__init__()
        self.add_include("iostream")
        self.add_include("fstream")


class STLAlgorithm(Class):
    def __init__(self):
        super().__init__()
        self.add_include("algorithm")
        self.add_include("functional")


class STLMeta(Class):
    def __init__(self):
        super().__init__()
        self.add_include("type_traits")


class STL(Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(STLContainers, STLMemory, STLIO, STLAlgorithm,
                            STLMeta)
