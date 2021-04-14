from pccm.core import Class 

class STLContainers(Class):
    def __init__(self):
        super().__init__()
        self.add_include("#include <vector>")
        self.add_include("#include <array>")
        self.add_include("#include <map>")
        self.add_include("#include <set>")
        self.add_include("#include <unordered_map>")
        self.add_include("#include <unordered_set>")

class STLMemory(Class):
    def __init__(self):
        super().__init__()
        self.add_include("#include <memory>")

class STLIO(Class):
    def __init__(self):
        super().__init__()
        self.add_include("#include <iostream>")
        self.add_include("#include <fstream>")

class STLAlgorithm(Class):
    def __init__(self):
        super().__init__()
        self.add_include("#include <algorithm>")
        self.add_include("#include <functional>")


class STLMeta(Class):
    def __init__(self):
        super().__init__()
        self.add_include("#include <type_traits>")

class STL(Class):
    def __init__(self):
        super().__init__()
        self.add_dependency(STLContainers, STLMemory, STLIO, STLAlgorithm, STLMeta)