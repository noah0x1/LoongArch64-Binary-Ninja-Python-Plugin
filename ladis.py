import re
import struct
from enum import IntEnum
from functools import lru_cache

try:
    from . import la64_opcodes
except ImportError:
    import la64_opcodes


GPR = ['$zero', '$ra', '$tp', '$sp',
       '$a0', '$a1', '$a2', '$a3', '$a4', '$a5', '$a6', '$a7',
       '$t0', '$t1', '$t2', '$t3', '$t4', '$t5', '$t6', '$t7', '$t8',
       '$r21', '$fp',
       '$s0', '$s1', '$s2', '$s3', '$s4', '$s5', '$s6', '$s7', '$s8']
FPR = (['$fa%d' % i for i in range(8)] +
       ['$ft%d' % i for i in range(16)] +
       ['$fs%d' % i for i in range(8)])
VR = ['$vr%d' % i for i in range(32)]
XR = ['$xr%d' % i for i in range(32)]
FCC = ['$fcc%d' % i for i in range(8)]
FCSR = ['$fcsr%d' % i for i in range(4)]
SCR = ['$scr%d' % i for i in range(4)]


class OPER_TYPE(IntEnum):
    GPR = 0
    FPR = 1
    VR = 2
    XR = 3
    FCC = 4
    FCSR = 5
    SCR = 6
    IMM = 7
    UIMM = 8
    ADDR = 9


REG_NAMES = {
    OPER_TYPE.GPR: GPR, OPER_TYPE.FPR: FPR, OPER_TYPE.VR: VR, OPER_TYPE.XR: XR,
    OPER_TYPE.FCC: FCC, OPER_TYPE.FCSR: FCSR, OPER_TYPE.SCR: SCR,
}


_REG_SPECS = {}
for _prefix, _t in (('r', OPER_TYPE.GPR), ('f', OPER_TYPE.FPR),
                    ('v', OPER_TYPE.VR), ('x', OPER_TYPE.XR)):
    for _suffix, _pos in (('d', 0), ('j', 5), ('k', 10), ('a', 15)):
        _REG_SPECS[_prefix + _suffix] = (_t, _pos, 5)
_REG_SPECS.update({
    'cd': (OPER_TYPE.FCC, 0, 3), 'cj': (OPER_TYPE.FCC, 5, 3), 'ca': (OPER_TYPE.FCC, 15, 3),
    'fcsrd': (OPER_TYPE.FCSR, 0, 2), 'fcsrj': (OPER_TYPE.FCSR, 5, 2),
    'scrd': (OPER_TYPE.SCR, 0, 2), 'scrj': (OPER_TYPE.SCR, 5, 2),
})

_IMM_RE = re.compile(r'^(ui|si)(\d+)(?:@(\d+))?(?:<<(\d+))?(?:\+(\d+))?$')


def sext(value, bits):
    sign = 1 << (bits - 1)
    value &= (1 << bits) - 1
    return (value ^ sign) - sign


def _split21(w):
    return sext(((w & 0x1f) << 16) | ((w >> 10) & 0xffff), 21)


def _split26(w):
    return sext(((w & 0x3ff) << 16) | ((w >> 10) & 0xffff), 26)


def _compile_spec(spec):
    if spec in _REG_SPECS:
        t, pos, width = _REG_SPECS[spec]
        mask = (1 << width) - 1
        return lambda w, a: (t, (w >> pos) & mask)
    if spec == 'off16':
        return lambda w, a: (OPER_TYPE.ADDR, (a + (sext(w >> 10, 16) << 2)) & 0xffffffffffffffff)
    if spec == 'off21':
        return lambda w, a: (OPER_TYPE.ADDR, (a + (_split21(w) << 2)) & 0xffffffffffffffff)
    if spec == 'off26':
        return lambda w, a: (OPER_TYPE.ADDR, (a + (_split26(w) << 2)) & 0xffffffffffffffff)
    if spec == 'sx21<<2':
        return lambda w, a: (OPER_TYPE.IMM, _split21(w) << 2)
    m = _IMM_RE.match(spec)
    if not m:
        raise ValueError('bad operand spec %r' % spec)
    kind, width, pos, shift, add = m.groups()
    width = int(width)
    pos = int(pos) if pos else 10
    shift = int(shift) if shift else 0
    add = int(add) if add else 0
    mask = (1 << width) - 1
    if kind == 'si':
        return lambda w, a: (OPER_TYPE.IMM, (sext(w >> pos, width) << shift) + add)
    return lambda w, a: (OPER_TYPE.UIMM, (((w >> pos) & mask) << shift) + add)


class OpcodeEntry:
    __slots__ = ('index', 'match', 'mask', 'name', 'specs', 'extractors')

    def __init__(self, index, match, mask, name, spec_str):
        self.index = index
        self.match = match
        self.mask = mask
        self.name = name
        self.specs = tuple(s for s in spec_str.split(',') if s)
        self.extractors = tuple(_compile_spec(s) for s in self.specs)


OPCODES = [OpcodeEntry(i, *row) for i, row in enumerate(la64_opcodes.ALL)]

_BY_MASK = {}
for _e in OPCODES:
    _BY_MASK.setdefault(_e.mask, {}).setdefault(_e.match, _e)
_MASK_TABLES = tuple(_BY_MASK.items())


@lru_cache(maxsize=1 << 16)
def lookup(word):
    best = None
    for mask, table in _MASK_TABLES:
        e = table.get(word & mask)
        if e is not None and (best is None or e.index < best.index):
            best = e
    return best


class Instruction:
    __slots__ = ('addr', 'word', 'name', 'specs', 'operands', 'length')

    def __init__(self, addr, word, entry):
        self.addr = addr
        self.word = word
        self.name = entry.name
        self.specs = entry.specs
        self.operands = [ex(word, addr) for ex in entry.extractors]
        self.length = 4

    def __repr__(self):
        return '<%s %s>' % (self.name, self.operands)


def decode(data, addr):
    if len(data) < 4:
        return None
    word = struct.unpack('<I', data[:4])[0]
    entry = lookup(word)
    if entry is None:
        return None
    return Instruction(addr, word, entry)


def alias(insn):
    n, ops, w = insn.name, insn.operands, insn.word
    if n == 'andi' and w == 0x03400000:
        return 'nop', []
    if n == 'or' and ops[2][1] == 0:
        return 'move', ops[:2]
    if n in ('addi.w', 'addi.d', 'ori') and ops[1][1] == 0:
        return ('li.d' if n == 'addi.d' else 'li.w'), [ops[0], ops[2]]
    if n == 'jirl' and ops[0][1] == 0 and ops[2][1] == 0:
        if ops[1][1] == 1:
            return 'ret', []
        return 'jr', [ops[1]]
    if n in ('blt', 'bge'):
        if ops[1][1] == 0:
            return n + 'z', [ops[0], ops[2]]
        if ops[0][1] == 0:
            return ('bgtz' if n == 'blt' else 'blez'), [ops[1], ops[2]]
    return n, ops
