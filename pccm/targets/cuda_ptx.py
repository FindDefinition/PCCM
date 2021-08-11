import abc
import contextlib
import enum
from typing import Dict, List, Optional, Set, Tuple, Union

from ccimport import compat

from pccm.core import Argument, FunctionCode
from pccm.utils import UniqueNamePool


class PointerType(enum.Enum):
    Global = "Global"
    Smem = "Smem"
    ExternalRegister = "ExternalRegister"


class RegDType(enum.Enum):
    F16 = "f16,16"
    F16X2 = "f16x2,32"
    F32 = "f32,32"
    F64 = "f64,64"
    S8 = "s8,8"
    S16 = "s16,16"
    S32 = "s32,32"
    S64 = "s64,64"
    U8 = "u8,8"
    U16 = "u16,16"
    U32 = "u32,32"
    U64 = "u64,64"
    B8 = "b8,8"
    B16 = "b16,16"
    B32 = "b32,32"
    B64 = "b64,64"
    BF16 = "bf16,16"
    TF32 = "tf32,32"
    PRED = "pred,1"


class OperandBase:
    def __init__(self, name: str):
        self.is_input = False
        self.is_output = False
        self.index = -1
        self.offset = 0
        self.name = name

    def __hash__(self):
        return hash(self.name)


class PointerBase(OperandBase):
    def __init__(self, name: str, ptr_type: PointerType):
        super().__init__(name)
        self.ptr_type = ptr_type

    def get_inline_asm_dtype(self):
        # all pointers are 64bit EXCEPT smem
        return "l" if self.ptr_type == PointerType.Global else "r"


class Pointer(PointerBase):
    def __init__(self, name: str, ptr_type: PointerType):
        super().__init__(name, ptr_type)

    def copy(self):
        res = Pointer(self.name, self.ptr_type)
        res.is_input = self.is_input
        res.is_output = self.is_output
        res.index = self.index
        res.offset = self.offset
        return res

    def __add__(self, offset: int):
        assert self.ptr_type != PointerType.ExternalRegister
        res = self.copy()
        res.offset += offset
        return res


class RegisterBase(OperandBase):
    def __init__(self, name: str, dtype: RegDType, count: int = 1):
        super().__init__(name)
        self.dtype = dtype
        dtype_value: List[str] = dtype.value.split(",")
        self.dtype_str = dtype_value[0]
        self.bitsize = int(dtype_value[1])
        self.count = count
        self.is_addr_register = False
        assert count >= 1 and count <= 4

    def __hash__(self):
        return hash(self.name)


class Register(RegisterBase):
    def decl_stmt(self) -> str:
        if self.count > 1:
            return f".reg .v{self.count} .{self.dtype_str} {self.name};"
        else:
            return f".reg .{self.dtype_str} {self.name};"

    def copy(self):
        res = Register(self.name, self.dtype, self.count)
        res.is_input = self.is_input
        res.is_output = self.is_output
        res.index = self.index
        res.offset = self.offset

        res.is_addr_register = self.is_addr_register
        # other attrs are immutable.
        return res

    def __add__(self, offset: int):
        res = self.copy()
        res.offset += offset
        return res

    def as_addr_reg(self):
        res = self.copy()
        res.is_addr_register = True
        return res

    def as_output_reg(self):
        res = self.copy()
        res.is_output = True
        return res


class ExternalRegisterPointer(PointerBase):
    kAllowedExtDTypesToLetter = {
        RegDType.U16: "h",
        RegDType.U32: "r",
        RegDType.S32: "r",
        RegDType.B32: "r",
        RegDType.U64: "l",
        RegDType.F32: "f",
        RegDType.F64: "d",
    }

    def __init__(self, name: str, dtype: RegDType):
        assert dtype in self.kAllowedExtDTypesToLetter
        super().__init__(name, PointerType.ExternalRegister)
        self.dtype = dtype
        dtype_value: List[str] = dtype.value.split(",")
        self.dtype_str = dtype_value[0]
        self.bitsize = int(dtype_value[1])
        self._inline_asm_constraint_letter = self.kAllowedExtDTypesToLetter[
            self.dtype]

    def get_inline_asm_dtype(self):
        return self._inline_asm_constraint_letter

    def __getitem__(self, offset: int):
        return ExternalRegister(self.name + "[{}]".format(offset), self.dtype)

    def unpack(self, start: int, stop: Optional[int] = None):
        if stop is None:
            stop = start
            start = 0
            assert stop > 0
        return [self[i] for i in range(start, stop)]


class ExternalRegister(RegisterBase):
    def __init__(self, name: str, dtype: RegDType):
        super().__init__(name, dtype)
        self._inline_asm_constraint_letter = ExternalRegisterPointer.kAllowedExtDTypesToLetter[
            self.dtype]

    def get_inline_asm_dtype(self):
        return self._inline_asm_constraint_letter

    def copy(self):
        res = ExternalRegister(self.name, self.dtype)
        res.is_input = self.is_input
        res.is_output = self.is_output
        res.index = self.index
        res.offset = self.offset

        res.is_addr_register = self.is_addr_register
        # other attrs are immutable.
        return res

    def __add__(self, offset: int):
        res = self.copy()
        res.offset += offset
        return res

    def as_addr_reg(self):
        res = self.copy()
        res.is_addr_register = True
        return res

    def as_output_reg(self):
        res = self.copy()
        res.is_output = True
        return res


PTX_PARAMS_TYPES_BASE = Union[Pointer, Register, ExternalRegister, str, int]

PTX_PARAMS_TYPES = Union[PTX_PARAMS_TYPES_BASE, List[PTX_PARAMS_TYPES_BASE]]

PTX_OPERAND_TYPES = Union[Pointer, ExternalRegister]
CONSTANT_TYPES = Union[str, int]
REG_INPUT_OPERAND_TYPES = Union[Register, ExternalRegister, str, int]
REG_OPERAND_TYPES = Union[Register, ExternalRegister]


class AsmStmt(abc.ABC):
    @abc.abstractmethod
    def get_stmt_str(self) -> str:
        return ""

    @abc.abstractmethod
    def get_operands(self) -> List[PTX_OPERAND_TYPES]:
        """childs must determine pointer i/o attr
        before call this method.
        """
        return []

    @abc.abstractmethod
    def get_registers(self) -> List[Register]:
        return []


class RegDeclStmt(AsmStmt):
    def __init__(self, name: str, dtype: RegDType, count: int = 1):
        self.reg = Register(name, dtype, count)

    def get_stmt_str(self) -> str:
        return ""

    def get_operands(self) -> List[PTX_OPERAND_TYPES]:
        """childs must determine pointer i/o attr
        before call this method.
        """
        return []

    def get_registers(self) -> List[Register]:
        return [self.reg]


class Generic(AsmStmt):
    def __init__(self,
                 stmt: str,
                 params: List[PTX_PARAMS_TYPES],
                 pred_reg: str = "",
                 has_output: bool = False):
        self.stmt = stmt
        self.params = params
        self.operands: List[PTX_OPERAND_TYPES] = []
        self.registers: List[Register] = []

        self.pred_reg = pred_reg
        self.has_output = has_output
        for p in params:
            if isinstance(p, list):
                for pp in p:
                    assert not isinstance(pp, ExternalRegisterPointer)
                    if isinstance(pp, (ExternalRegister, Pointer)):
                        self.operands.append(pp)
                    elif isinstance(pp, Register):
                        self.registers.append(pp)
            else:
                assert not isinstance(p, ExternalRegisterPointer)
                if isinstance(p, (ExternalRegister, Pointer)):
                    self.operands.append(p)
                elif isinstance(p, Register):
                    self.registers.append(p)

    def get_stmt_str(self) -> str:
        op_strs: List[str] = []
        name_to_op_str: Dict[str, str] = {}
        for op in self.params:
            if not isinstance(op, list):
                ops = [op]
            else:
                ops = op
            for op in ops:
                if isinstance(op, Pointer):
                    # addr
                    if op.offset > 0:
                        op_str = "[%{}+{}]".format(op.index, op.offset)
                    else:
                        op_str = "[%{}]".format(op.index)
                elif isinstance(op, ExternalRegister):
                    if op.is_addr_register:
                        if op.offset > 0:
                            op_str = "[%{}+{}]".format(op.index, op.offset)
                        else:
                            op_str = "[%{}]".format(op.index)
                    else:
                        op_str = "%{}".format(op.index)
                elif isinstance(op, Register):
                    if op.is_addr_register:
                        if op.offset > 0:
                            op_str = "[{}+{}]".format(op.name, op.offset)
                        else:
                            op_str = "[{}]".format(op.name)
                    else:
                        op_str = "{}".format(op.name)
                else:
                    op_str = "{}".format(op)
                if not isinstance(op, (str, int)):
                    name_to_op_str[op.name] = op_str

        for i, op in enumerate(self.params):
            if isinstance(op, list):
                if len(op) == 1:
                    if not isinstance(op[0], (str, int)):
                        op_str_with_packed = name_to_op_str[op[0].name]
                    else:
                        op_str_with_packed = str(op[0])
                else:
                    # op must be list of registers
                    # TODO check this.
                    # packed param
                    op_strs_regs: List[str] = []
                    for opp in op:
                        assert isinstance(
                            opp,
                            (Register,
                             ExternalRegister)) and not opp.is_addr_register
                        op_strs_regs.append(name_to_op_str[opp.name])
                    # if i == 0 and self.has_output:
                    #     op_str_with_packed = "|".join(op_strs_regs)
                    # else:
                    op_str_with_packed = "{{{}}}".format(
                        ", ".join(op_strs_regs))
            else:
                if not isinstance(op, (str, int)):
                    op_str_with_packed = name_to_op_str[op.name]
                else:
                    op_str_with_packed = str(op)
            op_strs.append(op_str_with_packed)
        if self.pred_reg:
            return "@{} {} {};".format(self.pred_reg, self.stmt,
                                       ",".join(op_strs))
        else:
            return "{} {};".format(self.stmt, ",".join(op_strs))

    def get_operands(self) -> List[PTX_OPERAND_TYPES]:
        """childs must determine pointer i/o attr
        before call this method.
        """
        return self.operands

    def get_registers(self) -> List[Register]:
        return self.registers


class CacheOpLd(enum.Enum):
    # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators
    AllLevel = "ca"
    L2AndBelow = "cg"
    Stream = "cs"
    LastUse = "lu"
    NoCache = "cv"


class CacheOpSt(enum.Enum):
    AllLevel = "wb"
    L2AndBelow = "cg"
    Stream = "cs"
    WriteThrough = "wt"


class PTXContext:
    """
    asm_stmt, operands (pointers and registers)
    for stmt in asm_stmts:
        ctx.query_decl("")
        ctx.assign_io(4) # io idx += 4
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions-setp

    TODO handle addr from 64bit register
    """
    def __init__(self):
        self._asm_stmts: List[AsmStmt] = []
        self._pre_asm_stmts: List[str] = []
        self._cur_if_pred = ""

    @contextlib.contextmanager
    def cond(self, pred_reg: Register, not_pred: bool = False):
        assert not self._cur_if_pred
        name = pred_reg.name
        if not_pred:
            name = "!" + pred_reg.name
        self._cur_if_pred = name
        yield pred_reg
        self._cur_if_pred = ""

    @contextlib.contextmanager
    def pred_if(self, name: str, cmp: str, a: REG_INPUT_OPERAND_TYPES,
                b: REG_INPUT_OPERAND_TYPES):
        assert not self._cur_if_pred
        pred_reg = Register(name, RegDType.PRED)
        self.setp(cmp, pred_reg, a, b)
        self._cur_if_pred = name
        yield pred_reg
        self._cur_if_pred = ""

    @staticmethod
    def reg(name: str, dtype: RegDType, count: int = 1):
        return Register(name, dtype, count)

    @staticmethod
    def ext_reg(name: str, dtype: RegDType):
        return ExternalRegister(name, dtype)

    @staticmethod
    def global_ptr(name: str):
        return Pointer(name, PointerType.Global)

    @staticmethod
    def smem_ptr(name: str):
        return Pointer(name, PointerType.Smem)

    @staticmethod
    def reg_ptr(name: str, dtype: RegDType):
        return ExternalRegisterPointer(name, dtype)

    def generic(self,
                stmt: str,
                params: List[PTX_PARAMS_TYPES],
                has_output: bool = False):

        self._asm_stmts.append(
            Generic(stmt, params, self._cur_if_pred, has_output))

    def _ld_or_st(self, is_ld: bool, addr: Pointer,
                  regs: Union[List[REG_OPERAND_TYPES], REG_OPERAND_TYPES],
                  cache_op: Optional[Union[CacheOpLd, CacheOpSt]]):
        if not addr.is_input:
            addr.is_input = is_ld
        if is_ld:
            if cache_op is not None:
                assert isinstance(cache_op, CacheOpLd)
        else:
            if cache_op is not None:
                assert isinstance(cache_op, CacheOpSt)

        if not isinstance(regs, list):
            regs = [regs]
        assert len(regs) in [1, 2, 4], "vec load only support 124"
        for reg in regs:
            if not reg.is_output:
                reg.is_output = is_ld
        ld_or_st = "ld" if is_ld else "st"
        if addr.ptr_type == PointerType.Global:
            stmt = "{}.global".format(ld_or_st)
        else:
            stmt = "{}.shared".format(ld_or_st)

        if len(regs) > 1:
            stmt += ".v{}".format(len(regs))
        if cache_op is not None:
            stmt += ".{}".format(cache_op.value)
        stmt += ".{}".format(regs[0].dtype_str)
        params: List[PTX_PARAMS_TYPES] = []
        if is_ld:
            params.append(list(regs))
            params.append(addr)
        else:
            params.append(addr)
            params.append(list(regs))
        self.generic(stmt, params, True)

    def setp(self, cmp_op: str, dst: REG_OPERAND_TYPES,
             a: REG_INPUT_OPERAND_TYPES, b: REG_INPUT_OPERAND_TYPES):
        dst.is_output = True
        dtype = ""
        if not isinstance(a, (str, int)):
            a.is_input = True
            dtype = a.dtype_str
        if not isinstance(b, (str, int)):
            b.is_input = True
            dtype = b.dtype_str
        assert dtype, "a or b must at least reg"
        stmt = "setp.{}.{}".format(cmp_op, dtype)
        self.generic(stmt, [dst, a, b], True)

    def ld(self,
           addr: Pointer,
           regs: Union[List[REG_OPERAND_TYPES], REG_OPERAND_TYPES],
           cache_op: Optional[CacheOpLd] = None):
        return self._ld_or_st(True, addr, regs, cache_op)

    def st(self,
           addr: Pointer,
           regs: Union[List[REG_OPERAND_TYPES], REG_OPERAND_TYPES],
           cache_op: Optional[CacheOpSt] = None):
        return self._ld_or_st(False, addr, regs, cache_op)

    def mov(self, dst: REG_OPERAND_TYPES,
            srcs: Union[List[REG_INPUT_OPERAND_TYPES],
                        REG_INPUT_OPERAND_TYPES]):
        if not isinstance(srcs, list):
            srcs_check = [srcs]
        else:
            srcs_check = srcs
        for src in srcs_check:
            if not isinstance(src, (str, int)):
                src.is_input = True
        dst.is_output = True

        assert len(srcs_check) in [1, 2, 4], "vec load only support 124"
        params: List[PTX_PARAMS_TYPES] = []
        params.append(dst)
        params.append(srcs)
        self._asm_stmts.append(
            Generic("mov.{}".format(dst.dtype_str), params, self._cur_if_pred))

    def _get_str(self) -> str:
        ptr_idx = 0
        stmt_strs: List[str] = []
        name_to_operands: Dict[str, List[PTX_OPERAND_TYPES]] = {}
        all_outputs: List[PTX_OPERAND_TYPES] = []
        all_inputs: List[PTX_OPERAND_TYPES] = []
        # we need to group operands by name to
        # set index correctly.
        for stmt in self._asm_stmts:
            operands = stmt.get_operands()
            for op in operands:
                # print(op.is_input, op.is_output, op, op.name)
                if op.name not in name_to_operands:
                    name_to_operands[op.name] = []
                name_to_operands[op.name].append(op)
        for _, vs in name_to_operands.items():
            # reduce op setting with same name
            is_out = False
            is_inp = False
            leader = vs[0]
            for v in vs:
                if v.is_output:
                    is_out = True
                if v.is_input:
                    is_inp = True
            leader.is_output = is_out
            leader.is_input = is_inp
            if not leader.is_output:
                all_inputs.append(leader)
            else:
                all_outputs.append(leader)

        for op in all_outputs:
            for op_same_name in name_to_operands[op.name]:
                op_same_name.index = ptr_idx
            ptr_idx += 1
        for op in all_inputs:
            for op_same_name in name_to_operands[op.name]:
                op_same_name.index = ptr_idx
            ptr_idx += 1
        # after set index of ptrs, we can get real asm stmt.
        all_regs: Dict[str, Register] = {}
        for stmt in self._asm_stmts:
            regs = stmt.get_registers()
            for r in regs:
                if r.name not in all_regs:
                    all_regs[r.name] = r
                    stmt_strs.append(r.decl_stmt())
            stmt_str = stmt.get_stmt_str()
            if stmt_str:
                stmt_strs.append(stmt_str)
        asm_output_strs: List[str] = []
        asm_input_strs: List[str] = []
        # TODO check all ops with same name, they must have same type.
        for op in all_outputs:
            op_has_input = False
            for op_same_name in name_to_operands[op.name]:
                if op_same_name.is_input:
                    op_has_input = True
                    break
            if op_has_input:
                constraint_letter = "+"
            else:
                constraint_letter = "="
            asm_output_strs.append("\"{}{}\"({})".format(
                constraint_letter, op.get_inline_asm_dtype(), op.name))
        for op in all_inputs:
            asm_input_strs.append("\"{}\"({})".format(
                op.get_inline_asm_dtype(), op.name))
        outputs_str = ", ".join(asm_output_strs)
        inputs_str = ", ".join(asm_input_strs)
        stmt_prefix = "            "
        asm_stmts_str = "\n".join(
            [stmt_prefix + "\"  {}\\n\"".format(s) for s in stmt_strs])
        if len(asm_output_strs) == 0:
            if len(asm_input_strs) == 0 and len(stmt_strs) == 0:
                return ""
            return """
            asm volatile (
            "{{\\n"
{asm_stmts_str}
            "}}\\n"
            :: {inputs}
            );
            """.format(asm_stmts_str=asm_stmts_str, inputs=inputs_str)

        return """
        asm volatile (
            "{{\\n"
{asm_stmts_str}
            "}}\\n"
            : {outputs}
            : {inputs}
        );
        """.format(asm_stmts_str=asm_stmts_str,
                   outputs=outputs_str,
                   inputs=inputs_str)


class PTXCode(FunctionCode):
    """TODO
    A = code.global_ptr("", length=...)
    B = code.smem_ptr("", length=...)
    C = code.register_ptr("", length=...)
    with code.asm_if("var_name"):
        a, b, c = code.decl_reg("a, b, c")
        code.asm_assign(A + 8, B + 16)
        code.asm_assign(a, 0)

    """
    def __init__(self,
                 code: str = "",
                 arguments: Optional[List[Argument]] = None,
                 return_type: str = "void",
                 ctor_inits: Optional[List[Tuple[str, str]]] = None):
        super().__init__(code, arguments, return_type, ctor_inits)
        self._asm_mode = False

    @contextlib.contextmanager
    def asm_block(self):
        assert not self._asm_mode, "you can't enter asm block twice or more."
        ctx = PTXContext()
        self._asm_mode = True
        yield ctx
        self._asm_mode = False
        res = ctx._get_str()
        # print(res)
        self.raw(res)
        # raise NotImplementedError


if __name__ == "__main__":
    ctx = PTXContext()
    # ptr_g = ctx.global_ptr("A") + 4
    # ptr_er2 = ctx.reg_ptr("fragA", RegDType.U64)

    # ptr_er0 = ctx.reg_ptr("frag", RegDType.U32)
    # ctx.ld(ptr_g, [ptr_er0[0], ptr_er0[1]])
    # ctx.st(ptr_g, [ptr_er0[0], ptr_er0[1]])

    # ctx.mov(ptr_er0[0], ptr_er0[1])
    # ctx.mov(ptr_er2[0], ptr_er0.unpack(4))
    # with ctx.pred_if("p", "ne", ptr_er0[0], 0):
    #     ctx.mov(ptr_er0[0], ptr_er0[1])
    # ctx.mov(ptr_er0[0], ptr_er0[1])

    ptr_addr = ctx.global_ptr("addr")
    frag = ctx.reg_ptr("frag", RegDType.U32)

    pred = ctx.ext_reg("(int)pred", RegDType.U32)
    for i in range(8):
        ctx.mov(frag[i], 0)
    with ctx.pred_if("p", "ne", pred, 0):
        frag_unpack = frag.unpack(8)
        ctx.ld(ptr_addr, frag_unpack[:4])
        ctx.ld(ptr_addr + 8 * 4, frag_unpack[4:])

    print(ctx._get_str())
