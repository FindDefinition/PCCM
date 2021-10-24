# Copyright 2021 Yan Yan
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
# build a class
pccm pkg.mod1.mod2.class0 -o lib_name --build_dir ...
# build a param class
pccm "pkg.mod1.pclass[a=5, b=3, x=y]" -o lib_name --build_dir ...
# generate code and ninja build file
pccm-gen pkg.mod1.mod2.class0 xxx_folder 
pccm-gen "pkg.mod1.pclass[a=5, b=3, x=y]" xxx_folder 
# generate code and cmake files
pccm-gen-cmake pkg.mod1.mod2.class0 xxx_folder 
"""

import sys 
import importlib

def import_name(name, package=None):
    module = importlib.import_module(name, package)
    return module


def main_gen():
    pass

if __name__ == "__main__":
    print(sys.argv)