from binaryninja import (Architecture, RegisterInfo, InstructionInfo, InstructionTextToken,
                         IntrinsicInfo, Type, CallingConvention, Platform, BinaryViewType,
                         PluginCommand)
from binaryninja.enums import InstructionTextTokenType, BranchType, Endianness #text
from binaryninja.log import log_info, log_error

from . import ladis, la64il, plt
from .ladis import OPER_TYPE, REG_NAMES, GPR, FPR

SHOW_ALIASES = True
MNEMONIC_WIDTH = 12

M64 = 0xffffffffffffffff
COND_BRANCHES = ('beq', 'bne', 'blt', 'bge', 'bltu', 'bgeu', 'beqz', 'bnez', 'bceqz', 'bcnez')
PC_RELATIVE = ('pcaddi', 'pcaddu12i', 'pcaddu18i', 'pcalau12i')


def _build_regs():
    regs = {}
    for name in GPR:
        regs[name] = RegisterInfo(name, 8)
    for i in range(32):
        xr = '$xr%d' % i
        regs[xr] = RegisterInfo(xr, 32)
        regs['$vr%d' % i] = RegisterInfo(xr, 16, 0)
        regs[FPR[i]] = RegisterInfo(xr, 8, 0)
        regs[FPR[i] + '_s'] = RegisterInfo(xr, 4, 0)
    for i in range(8):
        regs['$fcc%d' % i] = RegisterInfo('$fcc%d' % i, 1)
    for i in range(4):
        regs['$fcsr%d' % i] = RegisterInfo('$fcsr%d' % i, 4)
        regs['$scr%d' % i] = RegisterInfo('$scr%d' % i, 8)
    regs['eflags'] = RegisterInfo('eflags', 4)
    regs['ftop'] = RegisterInfo('ftop', 4)
    regs['$pc'] = RegisterInfo('$pc', 8)
    return regs


def _build_intrinsics():
    types = {}

    def t(size):
        if size not in types:
            types[size] = Type.int(size, False)
        return types[size]

    return {name: IntrinsicInfo([t(s) for s in ins], [t(s) for s in outs])
            for name, (ins, outs) in la64il.intrinsic_table().items()}


def _imm_text(t, v):
    if t == OPER_TYPE.UIMM:
        return '%#x' % v
    if -4096 < v < 4096:
        return '%d' % v
    return '-%#x' % -v if v < 0 else '%#x' % v


class LoongArch64(Architecture):
    name = 'LoongArch64'

    address_size = 8
    default_int_size = 4
    instr_alignment = 4
    max_instr_length = 4
    opcode_display_length = 4
    endianness = Endianness.LittleEndian

    regs = _build_regs()
    stack_pointer = '$sp'
    link_reg = '$ra'
    global_regs = ['$tp']


    intrinsics = _build_intrinsics()


    def get_instruction_info(self, data, addr):
        insn = ladis.decode(data, addr)
        if insn is None:
            return None

        result = InstructionInfo()
        result.length = 4
        name, ops = insn.name, insn.operands

        if name in COND_BRANCHES:
            result.add_branch(BranchType.TrueBranch, ops[-1][1])
            result.add_branch(BranchType.FalseBranch, addr + 4)
        elif name == 'b':
            result.add_branch(BranchType.UnconditionalBranch, ops[0][1])
        elif name == 'bl':
            result.add_branch(BranchType.CallDestination, ops[0][1])
        elif name == 'jirl':
            rd, rj, off = ops[0][1], ops[1][1], ops[2][1]
            if rd == 1:
                pass
            elif rd == 0 and rj == 1 and off == 0:
                result.add_branch(BranchType.FunctionReturn)
            else:
                result.add_branch(BranchType.IndirectBranch)
        elif name == 'break' and ops[0][1] in la64il.NORETURN_BREAK_CODES:
            result.add_branch(BranchType.ExceptionBranch)
        elif name == 'ertn':
            result.add_branch(BranchType.FunctionReturn)

        return result


    def get_instruction_text(self, data, addr):
        insn = ladis.decode(data, addr)
        if insn is None:
            return None

        name, ops = ladis.alias(insn) if SHOW_ALIASES else (insn.name, insn.operands)

        result = [InstructionTextToken(InstructionTextTokenType.InstructionToken, name)]
        if ops:
            result.append(InstructionTextToken(InstructionTextTokenType.TextToken,
                                               ' ' * max(1, MNEMONIC_WIDTH - len(name))))

        for i, (oper_type, oper_val) in enumerate(ops):
            if i:
                result.append(InstructionTextToken(InstructionTextTokenType.OperandSeparatorToken, ', '))
            if oper_type in REG_NAMES:
                result.append(InstructionTextToken(InstructionTextTokenType.RegisterToken,
                                                   REG_NAMES[oper_type][oper_val]))
            elif oper_type == OPER_TYPE.ADDR:
                result.append(InstructionTextToken(InstructionTextTokenType.PossibleAddressToken,
                                                   '%#x' % oper_val, oper_val))
            else:
                result.append(InstructionTextToken(InstructionTextTokenType.IntegerToken,
                                                   _imm_text(oper_type, oper_val), oper_val & M64))

        if insn.name in PC_RELATIVE:
            value = la64il.pcrel_value(insn)
            result.append(InstructionTextToken(InstructionTextTokenType.TextToken, '  # '))
            result.append(InstructionTextToken(InstructionTextTokenType.PossibleAddressToken,
                                               '%#x' % value, value))

        return result, 4


    def get_instruction_low_level_il(self, data, addr, il):
        insn = ladis.decode(data, addr)
        if insn is None:
            return None
        la64il.lift(self, insn, il)
        return 4


class LoongArch64CallingConvention(CallingConvention):
    name = 'lp64d'
    int_arg_regs = ['$a%d' % i for i in range(8)]
    float_arg_regs = ['$fa%d' % i for i in range(8)]
    int_return_reg = '$a0'
    high_int_return_reg = '$a1'
    float_return_reg = '$fa0'
    caller_saved_regs = (['$ra'] + ['$a%d' % i for i in range(8)] + ['$t%d' % i for i in range(9)] +
                         ['$fa%d' % i for i in range(8)] + ['$ft%d' % i for i in range(16)])
    callee_saved_regs = ['$s%d' % i for i in range(9)] + ['$fp'] + ['$fs%d' % i for i in range(8)]


class LoongArch64LinuxSyscall(CallingConvention):
    name = 'linux-syscall'
    int_arg_regs = ['$a7'] + ['$a%d' % i for i in range(7)]
    int_return_reg = '$a0'
    caller_saved_regs = ['$a0'] + ['$t%d' % i for i in range(9)]
    eligible_for_heuristics = False


class LinuxLoongArch64Platform(Platform):
    name = 'linux-loongarch64'


EM_LOONGARCH = 258

LoongArch64.register()
arch = Architecture['LoongArch64']

default_cc = LoongArch64CallingConvention(arch, 'lp64d')
syscall_cc = LoongArch64LinuxSyscall(arch, 'linux-syscall')
arch.register_calling_convention(default_cc)
arch.register_calling_convention(syscall_cc)
arch.standalone_platform.default_calling_convention = default_cc

linux = LinuxLoongArch64Platform(arch)
linux.default_calling_convention = default_cc
linux.system_call_convention = syscall_cc
linux.register('linux')

try:
    elf = BinaryViewType['ELF']
    elf.register_arch(EM_LOONGARCH, Endianness.LittleEndian, arch)
    elf.register_platform(0, arch, linux)
    elf.register_platform(3, arch, linux)
except Exception as e:
    log_error('LoongArch64: ELF registration failed: %s' % e)


def _is_la64(bv):
    return bv.arch is not None and bv.arch.name == LoongArch64.name


def _name_plt(bv):
    try:
        if _is_la64(bv) and bv.view_type == 'ELF':
            n = plt.name_plt_stubs(bv)
            if n:
                log_info('LoongArch64: named %d PLT stubs' % n)
    except Exception as e:
        log_error('LoongArch64: PLT naming failed: %s' % e)


PluginCommand.register('LoongArch64\\Name PLT stubs',
                       'Name PLT stubs of a dynamically linked LoongArch64 ELF',
                       _name_plt, _is_la64)

try:
    from binaryninja.binaryview import BinaryViewEvent
    from binaryninja.enums import BinaryViewEventType
    BinaryViewEvent.register(BinaryViewEventType.BinaryViewFinalizationEvent, _name_plt)
except Exception:
    try:
        BinaryViewType.add_binaryview_finalized_event(_name_plt)
    except Exception:
        log_info('LoongArch64: use "Plugins > LoongArch64 > Name PLT stubs" for imports')

log_info('LoongArch64 architecture loaded (%d opcodes)' % len(ladis.OPCODES))
