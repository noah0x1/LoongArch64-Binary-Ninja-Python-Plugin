import re

from binaryninja.lowlevelil import LowLevelILLabel, LLIL_TEMP

from .ladis import OPER_TYPE, GPR, FPR, VR, XR, FCC, FCSR, SCR, OPCODES

M64 = 0xffffffffffffffff

NORETURN_BREAK_CODES = (0, 6, 7)


def fpr_name(n, size):
    return FPR[n] + '_s' if size == 4 else FPR[n]


def _vals(insn):
    return [v for (_, v) in insn.operands]


def _r(il, n):
    return il.const(8, 0) if n == 0 else il.reg(8, GPR[n])


def _r32(il, n):
    return il.const(4, 0) if n == 0 else il.low_part(4, il.reg(8, GPR[n]))


def _rn(il, n, size):
    if size == 8:
        return _r(il, n)
    return il.const(size, 0) if n == 0 else il.low_part(size, il.reg(8, GPR[n]))


def _set(il, n, expr):
    if n == 0:
        il.append(il.nop())
    else:
        il.append(il.set_reg(8, GPR[n], expr))


def _setw(il, n, expr32):
    _set(il, n, il.sign_extend(8, expr32))


def _f(il, n, size):
    return il.reg(size, fpr_name(n, size))


def _fset(il, n, size, expr):
    il.append(il.set_reg(size, fpr_name(n, size), expr))


def _fsize(name):
    return 4 if name.endswith('.s') else 8


_LIFTERS = {}


def _lifts(*names):
    def deco(fn):
        for n in names:
            _LIFTERS[n] = fn
        return fn
    return deco


def lift(arch, insn, il):
    _LIFTERS.get(insn.name, lift_intrinsic)(arch, insn, il)


_OP3 = {
    'add.w': ('add', True), 'add.d': ('add', False),
    'sub.w': ('sub', True), 'sub.d': ('sub', False),
    'and': ('and_expr', False), 'or': ('or_expr', False), 'xor': ('xor_expr', False),
    'mul.w': ('mult', True), 'mul.d': ('mult', False),
    'div.w': ('div_signed', True), 'div.d': ('div_signed', False),
    'div.wu': ('div_unsigned', True), 'div.du': ('div_unsigned', False),
    'mod.w': ('mod_signed', True), 'mod.d': ('mod_signed', False),
    'mod.wu': ('mod_unsigned', True), 'mod.du': ('mod_unsigned', False),
}


@_lifts(*_OP3)
def _op3(arch, insn, il):
    rd, rj, rk = _vals(insn)
    op, is32 = _OP3[insn.name]
    if insn.name == 'or' and rk == 0:
        _set(il, rd, _r(il, rj))
    elif is32:
        _setw(il, rd, getattr(il, op)(4, _r32(il, rj), _r32(il, rk)))
    else:
        _set(il, rd, getattr(il, op)(8, _r(il, rj), _r(il, rk)))


@_lifts('nor', 'andn', 'orn')
def _op3_not(arch, insn, il):
    rd, rj, rk = _vals(insn)
    if insn.name == 'nor':
        v = il.not_expr(8, il.or_expr(8, _r(il, rj), _r(il, rk)))
    elif insn.name == 'andn':
        v = il.and_expr(8, _r(il, rj), il.not_expr(8, _r(il, rk)))
    else:
        v = il.or_expr(8, _r(il, rj), il.not_expr(8, _r(il, rk)))
    _set(il, rd, v)


@_lifts('slt', 'sltu', 'slti', 'sltui')
def _slt(arch, insn, il):
    rd, rj, x = _vals(insn)
    b = _r(il, x) if insn.specs[2] == 'rk' else il.const(8, x)
    if insn.name in ('slt', 'slti'):
        cmp = il.compare_signed_less_than(8, _r(il, rj), b)
    else:
        cmp = il.compare_unsigned_less_than(8, _r(il, rj), b)
    _set(il, rd, il.bool_to_int(8, cmp))


@_lifts('maskeqz', 'masknez')
def _mask(arch, insn, il):
    rd, rj, rk = _vals(insn)
    if insn.name == 'maskeqz':
        cond = il.compare_not_equal(8, _r(il, rk), il.const(8, 0))
    else:
        cond = il.compare_equal(8, _r(il, rk), il.const(8, 0))
    _set(il, rd, il.and_expr(8, _r(il, rj), il.neg_expr(8, il.bool_to_int(8, cond))))


@_lifts('addi.w')
def _addi_w(arch, insn, il):
    rd, rj, imm = _vals(insn)
    if rj == 0:
        _set(il, rd, il.const(8, imm))
    else:
        _setw(il, rd, il.add(4, _r32(il, rj), il.const(4, imm)))


@_lifts('addi.d', 'addu16i.d')
def _addi_d(arch, insn, il):
    rd, rj, imm = _vals(insn)
    if insn.name == 'addu16i.d':
        imm <<= 16
    if rj == 0:
        _set(il, rd, il.const(8, imm))
    elif imm == 0:
        _set(il, rd, _r(il, rj))
    else:
        _set(il, rd, il.add(8, _r(il, rj), il.const(8, imm)))


@_lifts('andi', 'ori', 'xori')
def _logic_imm(arch, insn, il):
    rd, rj, imm = _vals(insn)
    if rj == 0 and insn.name != 'andi':
        _set(il, rd, il.const(8, imm))
    else:
        op = {'andi': il.and_expr, 'ori': il.or_expr, 'xori': il.xor_expr}[insn.name]
        _set(il, rd, op(8, _r(il, rj), il.const(8, imm)))


@_lifts('lu12i.w')
def _lu12i(arch, insn, il):
    rd, imm = _vals(insn)
    _set(il, rd, il.const(8, imm << 12))


@_lifts('lu32i.d')
def _lu32i(arch, insn, il):
    rd, imm = _vals(insn)
    low = il.and_expr(8, _r(il, rd), il.const(8, 0xffffffff))
    _set(il, rd, il.or_expr(8, low, il.const(8, (imm << 32) & M64)))


@_lifts('lu52i.d')
def _lu52i(arch, insn, il):
    rd, rj, imm = _vals(insn)
    hi = (imm << 52) & M64
    if rj == 0:
        _set(il, rd, il.const(8, hi))
    else:
        low = il.and_expr(8, _r(il, rj), il.const(8, 0x000fffffffffffff))
        _set(il, rd, il.or_expr(8, low, il.const(8, hi)))


def pcrel_value(insn):
    imm = insn.operands[1][1]
    pc = insn.addr
    if insn.name == 'pcaddi':
        return (pc + (imm << 2)) & M64
    if insn.name == 'pcaddu12i':
        return (pc + (imm << 12)) & M64
    if insn.name == 'pcaddu18i':
        return (pc + (imm << 18)) & M64
    return (pc + (imm << 12)) & ~0xfff & M64


@_lifts('pcaddi', 'pcaddu12i', 'pcaddu18i', 'pcalau12i')
def _pcrel(arch, insn, il):
    _set(il, insn.operands[0][1], il.const_pointer(8, pcrel_value(insn)))


_SHIFT = {'sll': 'shift_left', 'srl': 'logical_shift_right',
          'sra': 'arith_shift_right', 'rotr': 'rotate_right'}


@_lifts('sll.w', 'srl.w', 'sra.w', 'rotr.w', 'sll.d', 'srl.d', 'sra.d', 'rotr.d')
def _shift_reg(arch, insn, il):
    rd, rj, rk = _vals(insn)
    base, width = insn.name.split('.')
    op = getattr(il, _SHIFT[base])
    if width == 'w':
        amount = il.and_expr(8, _r(il, rk), il.const(8, 31))
        _setw(il, rd, op(4, _r32(il, rj), amount))
    else:
        amount = il.and_expr(8, _r(il, rk), il.const(8, 63))
        _set(il, rd, op(8, _r(il, rj), amount))


@_lifts('slli.w', 'srli.w', 'srai.w', 'rotri.w', 'slli.d', 'srli.d', 'srai.d', 'rotri.d')
def _shift_imm(arch, insn, il):
    rd, rj, sa = _vals(insn)
    base, width = insn.name.split('.')
    op = getattr(il, _SHIFT[base[:-1]])
    if width == 'w':
        v = _r32(il, rj) if sa == 0 else op(4, _r32(il, rj), il.const(1, sa))
        _setw(il, rd, v)
    else:
        _set(il, rd, _r(il, rj) if sa == 0 else op(8, _r(il, rj), il.const(1, sa)))


@_lifts('alsl.w', 'alsl.wu', 'alsl.d')
def _alsl(arch, insn, il):
    rd, rj, rk, sa = _vals(insn)
    if insn.name == 'alsl.d':
        v = il.shift_left(8, _r(il, rj), il.const(1, sa))
        _set(il, rd, v if rk == 0 else il.add(8, v, _r(il, rk)))
        return
    v = il.shift_left(4, _r32(il, rj), il.const(1, sa))
    if rk != 0:
        v = il.add(4, v, _r32(il, rk))
    if insn.name == 'alsl.w':
        _setw(il, rd, v)
    else:
        _set(il, rd, il.zero_extend(8, v))


@_lifts('bytepick.w', 'bytepick.d')
def _bytepick(arch, insn, il):
    rd, rj, rk, sa = _vals(insn)
    bits = 32 if insn.name == 'bytepick.w' else 64
    size = bits // 8
    if sa == 0:
        v = _rn(il, rk, size)
    else:
        v = il.or_expr(size,
                       il.shift_left(size, _rn(il, rk, size), il.const(1, 8 * sa)),
                       il.logical_shift_right(size, _rn(il, rj, size), il.const(1, bits - 8 * sa)))
    if size == 4:
        _setw(il, rd, v)
    else:
        _set(il, rd, v)


@_lifts('ext.w.b', 'ext.w.h')
def _ext(arch, insn, il):
    rd, rj = _vals(insn)
    size = 1 if insn.name == 'ext.w.b' else 2
    _set(il, rd, il.sign_extend(8, _rn(il, rj, size)))


@_lifts('bstrins.w', 'bstrins.d', 'bstrpick.w', 'bstrpick.d')
def _bstr(arch, insn, il):
    rd, rj, msb, lsb = _vals(insn)
    if msb < lsb:
        il.append(il.undefined())
        return
    size = 4 if insn.name.endswith('.w') else 8
    full = (1 << (size * 8)) - 1
    width = msb - lsb + 1
    field = (1 << width) - 1

    if insn.name.startswith('bstrpick'):
        if size == 8 and lsb == 0 and width in (8, 16, 32):
            _set(il, rd, il.zero_extend(8, _rn(il, rj, width // 8)))
            return
        v = _rn(il, rj, size)
        if lsb:
            v = il.logical_shift_right(size, v, il.const(1, lsb))
        if msb != size * 8 - 1:
            v = il.and_expr(size, v, il.const(size, field))
        if size == 4:
            _setw(il, rd, v)
        else:
            _set(il, rd, v)
        return

    mask = field << lsb
    src = _rn(il, rj, size)
    if lsb:
        src = il.shift_left(size, src, il.const(1, lsb))
    v = il.or_expr(size,
                   il.and_expr(size, _rn(il, rd, size), il.const(size, ~mask & full)),
                   il.and_expr(size, src, il.const(size, mask)))
    if size == 4:
        _setw(il, rd, v)
    else:
        _set(il, rd, v)


@_lifts('mulh.w', 'mulh.wu', 'mulw.d.w', 'mulw.d.wu', 'mulh.d', 'mulh.du')
def _mulx(arch, insn, il):
    rd, rj, rk = _vals(insn)
    n = insn.name
    if n in ('mulh.d', 'mulh.du'):
        ext = il.sign_extend if n == 'mulh.d' else il.zero_extend
        shr = il.arith_shift_right if n == 'mulh.d' else il.logical_shift_right
        prod = il.mult(16, ext(16, _r(il, rj)), ext(16, _r(il, rk)))
        _set(il, rd, il.low_part(8, shr(16, prod, il.const(1, 64))))
        return
    signed = n in ('mulh.w', 'mulw.d.w')
    ext = il.sign_extend if signed else il.zero_extend
    prod = il.mult(8, ext(8, _r32(il, rj)), ext(8, _r32(il, rk)))
    if n.startswith('mulw'):
        _set(il, rd, prod)
    elif signed:
        _set(il, rd, il.arith_shift_right(8, prod, il.const(1, 32)))
    else:
        _setw(il, rd, il.low_part(4, il.logical_shift_right(8, prod, il.const(1, 32))))


_SIZE = {'b': 1, 'h': 2, 'w': 4, 'd': 8, 'bu': 1, 'hu': 2, 'wu': 4}
_BOUNDED = ('ldgt', 'ldle', 'stgt', 'stle', 'fldgt', 'fldle', 'fstgt', 'fstle')


def _mem_addr(il, insn):
    v = _vals(insn)
    base = v[1]
    if len(v) == 2 or insn.name.startswith(_BOUNDED):
        return _r(il, base)
    if insn.specs[2] == 'rk':
        return il.add(8, _r(il, base), _r(il, v[2]))
    if base == 0:
        return il.const_pointer(8, v[2] & M64)
    if v[2] == 0:
        return _r(il, base)
    return il.add(8, _r(il, base), il.const(8, v[2]))


@_lifts(*['ld.' + s for s in ('b', 'h', 'w', 'd', 'bu', 'hu', 'wu')],
        *['ldx.' + s for s in ('b', 'h', 'w', 'd', 'bu', 'hu', 'wu')],
        *['ldgt.' + s for s in 'bhwd'], *['ldle.' + s for s in 'bhwd'],
        'ldptr.w', 'ldptr.d', 'll.w', 'll.d', 'llacq.w', 'llacq.d')
def _load(arch, insn, il):
    suffix = insn.name.split('.')[1]
    size = _SIZE[suffix]
    v = il.load(size, _mem_addr(il, insn))
    if size != 8:
        v = il.zero_extend(8, v) if suffix.endswith('u') else il.sign_extend(8, v)
    _set(il, insn.operands[0][1], v)


@_lifts(*['st.' + s for s in 'bhwd'], *['stx.' + s for s in 'bhwd'],
        *['stgt.' + s for s in 'bhwd'], *['stle.' + s for s in 'bhwd'],
        'stptr.w', 'stptr.d', 'sc.w', 'sc.d', 'screl.w', 'screl.d')
def _store(arch, insn, il):
    size = _SIZE[insn.name.split('.')[1]]
    rd = insn.operands[0][1]
    il.append(il.store(size, _mem_addr(il, insn), _rn(il, rd, size)))
    if insn.name.startswith(('sc.', 'screl.')):
        _set(il, rd, il.const(8, 1))


_AMO_OPS = {'swap': None, 'add': 'add', 'and': 'and_expr', 'or': 'or_expr', 'xor': 'xor_expr'}
_AMO_NAMES = [('am%s%s.%s' % (op, db, s)) for op in _AMO_OPS for db in ('', '_db') for s in 'bhwd']


@_lifts(*[n for n in _AMO_NAMES if any(e.name == n for e in OPCODES)])
def _amo(arch, insn, il):
    rd, rk, rj = _vals(insn)
    base, suffix = insn.name.split('.')
    op = _AMO_OPS[base[2:].replace('_db', '')]
    size = _SIZE[suffix]
    old = LLIL_TEMP(0)
    il.append(il.set_reg(size, old, il.load(size, _r(il, rj))))
    new = _rn(il, rk, size) if op is None else getattr(il, op)(size, il.reg(size, old), _rn(il, rk, size))
    il.append(il.store(size, _r(il, rj), new))
    if rd != 0:
        v = il.reg(size, old)
        _set(il, rd, v if size == 8 else il.sign_extend(8, v))


@_lifts('preld', 'preldx')
def _prefetch(arch, insn, il):
    il.append(il.nop())


def _cond_branch(arch, il, cond, target, fallthrough):
    t = il.get_label_for_address(arch, target)
    f = il.get_label_for_address(arch, fallthrough)
    new_t, new_f = t is None, f is None
    if new_t:
        t = LowLevelILLabel()
    if new_f:
        f = LowLevelILLabel()
    il.append(il.if_expr(cond, t, f))
    if new_t:
        il.mark_label(t)
        il.append(il.jump(il.const_pointer(8, target)))
    if new_f:
        il.mark_label(f)


_BCOND = {'beq': 'compare_equal', 'bne': 'compare_not_equal',
          'blt': 'compare_signed_less_than', 'bge': 'compare_signed_greater_equal',
          'bltu': 'compare_unsigned_less_than', 'bgeu': 'compare_unsigned_greater_equal'}


@_lifts(*_BCOND)
def _bcc(arch, insn, il):
    rj, rd, target = _vals(insn)
    cond = getattr(il, _BCOND[insn.name])(8, _r(il, rj), _r(il, rd))
    _cond_branch(arch, il, cond, target, insn.addr + 4)


@_lifts('beqz', 'bnez')
def _bccz(arch, insn, il):
    rj, target = _vals(insn)
    op = il.compare_equal if insn.name == 'beqz' else il.compare_not_equal
    _cond_branch(arch, il, op(8, _r(il, rj), il.const(8, 0)), target, insn.addr + 4)


@_lifts('bceqz', 'bcnez')
def _bcfp(arch, insn, il):
    cj, target = _vals(insn)
    op = il.compare_equal if insn.name == 'bceqz' else il.compare_not_equal
    _cond_branch(arch, il, op(1, il.reg(1, FCC[cj]), il.const(1, 0)), target, insn.addr + 4)


@_lifts('b')
def _b(arch, insn, il):
    target = insn.operands[0][1]
    label = il.get_label_for_address(arch, target)
    if label is not None:
        il.append(il.goto(label))
    else:
        il.append(il.jump(il.const_pointer(8, target)))


@_lifts('bl')
def _bl(arch, insn, il):
    il.append(il.call(il.const_pointer(8, insn.operands[0][1])))


@_lifts('jirl')
def _jirl(arch, insn, il):
    rd, rj, off = _vals(insn)
    dest = _r(il, rj) if off == 0 else il.add(8, _r(il, rj), il.const(8, off))
    if rd == 1:
        il.append(il.call(dest))
    elif rd == 0:
        if rj == 1 and off == 0:
            il.append(il.ret(dest))
        else:
            il.append(il.jump(dest))
    else:
        tmp = LLIL_TEMP(0)
        il.append(il.set_reg(8, tmp, dest))
        il.append(il.set_reg(8, GPR[rd], il.const_pointer(8, (insn.addr + 4) & M64)))
        il.append(il.jump(il.reg(8, tmp)))


@_lifts('syscall')
def _syscall(arch, insn, il):
    il.append(il.system_call())


@_lifts('break')
def _break(arch, insn, il):
    code = insn.operands[0][1]
    if code in NORETURN_BREAK_CODES:
        il.append(il.trap(code))
    else:
        lift_intrinsic(arch, insn, il)


@_lifts('ertn')
def _ertn(arch, insn, il):
    lift_intrinsic(arch, insn, il)
    il.append(il.no_ret())


@_lifts('fld.s', 'fld.d', 'fldx.s', 'fldx.d', 'fldgt.s', 'fldgt.d', 'fldle.s', 'fldle.d')
def _fload(arch, insn, il):
    size = _fsize(insn.name)
    _fset(il, insn.operands[0][1], size, il.load(size, _mem_addr(il, insn)))


@_lifts('fst.s', 'fst.d', 'fstx.s', 'fstx.d', 'fstgt.s', 'fstgt.d', 'fstle.s', 'fstle.d')
def _fstore(arch, insn, il):
    size = _fsize(insn.name)
    il.append(il.store(size, _mem_addr(il, insn), _f(il, insn.operands[0][1], size)))


_FARITH = {'fadd': 'float_add', 'fsub': 'float_sub', 'fmul': 'float_mult', 'fdiv': 'float_div'}


@_lifts(*[op + sfx for op in _FARITH for sfx in ('.s', '.d')])
def _farith(arch, insn, il):
    fd, fj, fk = _vals(insn)
    size = _fsize(insn.name)
    op = getattr(il, _FARITH[insn.name.split('.')[0]])
    _fset(il, fd, size, op(size, _f(il, fj, size), _f(il, fk, size)))


@_lifts(*[op + sfx for op in ('fmadd', 'fmsub', 'fnmadd', 'fnmsub') for sfx in ('.s', '.d')])
def _fma(arch, insn, il):
    fd, fj, fk, fa = _vals(insn)
    size = _fsize(insn.name)
    base = insn.name.split('.')[0]
    prod = il.float_mult(size, _f(il, fj, size), _f(il, fk, size))
    op = il.float_add if base.endswith('madd') else il.float_sub
    v = op(size, prod, _f(il, fa, size))
    if base.startswith('fnm'):
        v = il.float_neg(size, v)
    _fset(il, fd, size, v)


def _one(il, size):
    return il.float_const_single(1.0) if size == 4 else il.float_const_double(1.0)


@_lifts(*[op + sfx for op in ('fabs', 'fneg', 'fsqrt', 'fmov', 'frecip', 'frsqrt', 'frint')
          for sfx in ('.s', '.d')])
def _funary(arch, insn, il):
    fd, fj = _vals(insn)
    size = _fsize(insn.name)
    op = insn.name.split('.')[0]
    src = _f(il, fj, size)
    v = {'fabs': lambda: il.float_abs(size, src),
         'fneg': lambda: il.float_neg(size, src),
         'fsqrt': lambda: il.float_sqrt(size, src),
         'fmov': lambda: src,
         'frecip': lambda: il.float_div(size, _one(il, size), src),
         'frsqrt': lambda: il.float_div(size, _one(il, size), il.float_sqrt(size, src)),
         'frint': lambda: il.round_to_int(size, src)}[op]()
    _fset(il, fd, size, v)


@_lifts('movgr2fr.w', 'movgr2fr.d', 'movgr2frh.w', 'movfr2gr.s', 'movfr2gr.d', 'movfrh2gr.s',
        'movgr2fcsr', 'movfcsr2gr', 'movfr2cf', 'movcf2fr', 'movgr2cf', 'movcf2gr')
def _fmoves(arch, insn, il):
    d, s = _vals(insn)
    n = insn.name
    if n == 'movgr2fr.w':
        _fset(il, d, 4, _r32(il, s))
    elif n == 'movgr2fr.d':
        _fset(il, d, 8, _r(il, s))
    elif n == 'movgr2frh.w':
        hi = il.shift_left(8, il.zero_extend(8, _r32(il, s)), il.const(1, 32))
        _fset(il, d, 8, il.or_expr(8, hi, il.zero_extend(8, _f(il, d, 4))))
    elif n == 'movfr2gr.s':
        _set(il, d, il.sign_extend(8, _f(il, s, 4)))
    elif n == 'movfr2gr.d':
        _set(il, d, _f(il, s, 8))
    elif n == 'movfrh2gr.s':
        _setw(il, d, il.low_part(4, il.logical_shift_right(8, _f(il, s, 8), il.const(1, 32))))
    elif n == 'movgr2fcsr':
        il.append(il.set_reg(4, FCSR[d], _r32(il, s)))
    elif n == 'movfcsr2gr':
        _set(il, d, il.sign_extend(8, il.reg(4, FCSR[s])))
    elif n == 'movfr2cf':
        il.append(il.set_reg(1, FCC[d], il.and_expr(1, il.low_part(1, _f(il, s, 8)), il.const(1, 1))))
    elif n == 'movgr2cf':
        il.append(il.set_reg(1, FCC[d], il.and_expr(1, _rn(il, s, 1), il.const(1, 1))))
    elif n == 'movcf2fr':
        _fset(il, d, 8, il.zero_extend(8, il.reg(1, FCC[s])))
    else:
        _set(il, d, il.zero_extend(8, il.reg(1, FCC[s])))


@_lifts('fcvt.s.d', 'fcvt.d.s', 'ffint.s.w', 'ffint.s.l', 'ffint.d.w', 'ffint.d.l')
def _fconv(arch, insn, il):
    fd, fj = _vals(insn)
    _, dst_t, src_t = insn.name.split('.')
    dsize = 4 if dst_t in ('s', 'w') else 8
    ssize = 4 if src_t in ('s', 'w') else 8
    src = _f(il, fj, ssize)
    if insn.name.startswith('fcvt'):
        _fset(il, fd, dsize, il.float_convert(dsize, src))
    else:
        _fset(il, fd, dsize, il.int_to_float(dsize, src))


_FTINT_RE = re.compile(r'^ftint(rm|rp|rz|rne)?\.([wl])\.([sd])$')


@_lifts(*[e.name for e in OPCODES if _FTINT_RE.match(e.name)])
def _ftint(arch, insn, il):
    fd, fj = _vals(insn)
    mode, dst_t, src_t = _FTINT_RE.match(insn.name).groups()
    ssize = 4 if src_t == 's' else 8
    dsize = 4 if dst_t == 'w' else 8
    src = _f(il, fj, ssize)
    if mode == 'rm':
        src = il.floor(ssize, src)
    elif mode == 'rp':
        src = il.ceil(ssize, src)
    elif mode != 'rz':
        src = il.round_to_int(ssize, src)
    _fset(il, fd, dsize, il.float_to_int(dsize, src))


_FCMP_RE = re.compile(r'^fcmp\.[cs](\w+)\.([sd])$')


@_lifts(*[e.name for e in OPCODES if _FCMP_RE.match(e.name)])
def _fcmp(arch, insn, il):
    cd, fj, fk = _vals(insn)
    cond, fmt = _FCMP_RE.match(insn.name).groups()
    size = 4 if fmt == 's' else 8

    def cmp(op):
        return il.bool_to_int(1, getattr(il, 'float_compare_' + op)(size, _f(il, fj, size), _f(il, fk, size)))

    def inv(op):
        return il.xor_expr(1, cmp(op), il.const(1, 1))

    v = {'af': lambda: il.const(1, 0),
         'lt': lambda: cmp('less_than'),
         'eq': lambda: cmp('equal'),
         'le': lambda: cmp('less_equal'),
         'un': lambda: cmp('unordered'),
         'or': lambda: cmp('ordered'),
         'ult': lambda: inv('greater_equal'),
         'ule': lambda: inv('greater_than'),
         'une': lambda: cmp('not_equal'),
         'ueq': lambda: il.or_expr(1, cmp('equal'), cmp('unordered')),
         'ne': lambda: il.or_expr(1, cmp('less_than'), cmp('greater_than'))}[cond]()
    il.append(il.set_reg(1, FCC[cd], v))


@_lifts('fsel')
def _fsel(arch, insn, il):
    fd, fj, fk, ca = _vals(insn)
    t, f, done = LowLevelILLabel(), LowLevelILLabel(), LowLevelILLabel()
    il.append(il.if_expr(il.compare_not_equal(1, il.reg(1, FCC[ca]), il.const(1, 0)), t, f))
    il.mark_label(t)
    _fset(il, fd, 8, _f(il, fk, 8))
    il.append(il.goto(done))
    il.mark_label(f)
    _fset(il, fd, 8, _f(il, fj, 8))
    il.append(il.goto(done))
    il.mark_label(done)


@_lifts('vld', 'vldx', 'xvld', 'xvldx')
def _vload(arch, insn, il):
    size, regs = (32, XR) if insn.name.startswith('x') else (16, VR)
    il.append(il.set_reg(size, regs[insn.operands[0][1]], il.load(size, _mem_addr(il, insn))))


@_lifts('vst', 'vstx', 'xvst', 'xvstx')
def _vstore(arch, insn, il):
    size, regs = (32, XR) if insn.name.startswith('x') else (16, VR)
    il.append(il.store(size, _mem_addr(il, insn), il.reg(size, regs[insn.operands[0][1]])))


@_lifts('vrepli.b', 'vrepli.h', 'vrepli.w', 'vrepli.d',
        'xvrepli.b', 'xvrepli.h', 'xvrepli.w', 'xvrepli.d')
def _vrepli(arch, insn, il):
    vd, imm = _vals(insn)
    if imm != 0:
        lift_intrinsic(arch, insn, il)
        return
    size, regs = (32, XR) if insn.name.startswith('x') else (16, VR)
    il.append(il.set_reg(size, regs[vd], il.const(size, 0)))


_DEST_SPECS = ('rd', 'fd', 'vd', 'xd', 'cd', 'fcsrd', 'scrd')
_STORE_LIKE = ('st', 'fst', 'vst', 'xvst', 'iocsrwr')
_TWO_OUTPUTS = ('rdtimel.w', 'rdtimeh.w', 'rdtime.d')
_READS_DEST = re.compile(
    r'^(x?v(madd|msub|shuf\.[hwd]|shuf4i\.d|permi\.[wq]|extrins|bitseli|insgr2vr|frstp)|xvinsve0'
    r'|x?vs{1,2}(rl|ra)r?ni\.|g?csr(wr|xchg)|x86settag)')


def intrinsic_name(mnemonic):
    return '__' + mnemonic.replace('.', '_')


_SPEC_SIZE = {'rd': 8, 'rj': 8, 'rk': 8, 'ra': 8,
              'vd': 16, 'vj': 16, 'vk': 16, 'va': 16,
              'xd': 32, 'xj': 32, 'xk': 32, 'xa': 32,
              'cd': 1, 'cj': 1, 'ca': 1, 'fcsrd': 4, 'fcsrj': 4, 'scrd': 8, 'scrj': 8}


def _operand_size(mnemonic, spec):
    if spec in ('fd', 'fj', 'fk', 'fa'):
        return _fsize(mnemonic)
    return _SPEC_SIZE.get(spec, 8)


def intrinsic_signature(mnemonic, specs):
    if mnemonic in _TWO_OUTPUTS:
        return [0, 1], []
    if specs and specs[0] in _DEST_SPECS and not mnemonic.startswith(_STORE_LIKE):
        ins = list(range(1, len(specs)))
        if _READS_DEST.match(mnemonic):
            ins = [0] + ins
        return [0], ins
    return [], list(range(len(specs)))


def intrinsic_table():
    table = {}
    for e in OPCODES:
        outs, ins = intrinsic_signature(e.name, e.specs)
        name = intrinsic_name(e.name)
        assert name not in table or table[name] == ([_operand_size(e.name, e.specs[i]) for i in ins],
                                                     [_operand_size(e.name, e.specs[i]) for i in outs])
        table[name] = ([_operand_size(e.name, e.specs[i]) for i in ins],
                       [_operand_size(e.name, e.specs[i]) for i in outs])
    return table


def _reg_for(insn, i, size):
    t, v = insn.operands[i]
    if t == OPER_TYPE.FPR:
        return fpr_name(v, size)
    return {OPER_TYPE.GPR: GPR, OPER_TYPE.VR: VR, OPER_TYPE.XR: XR, OPER_TYPE.FCC: FCC,
            OPER_TYPE.FCSR: FCSR, OPER_TYPE.SCR: SCR}[t][v]


def lift_intrinsic(arch, insn, il):
    outs, ins = intrinsic_signature(insn.name, insn.specs)
    outputs = []
    for i in outs:
        t, v = insn.operands[i]
        if t == OPER_TYPE.GPR and v == 0:
            outputs.append(LLIL_TEMP(i))
        else:
            outputs.append(_reg_for(insn, i, _operand_size(insn.name, insn.specs[i])))
    params = []
    for i in ins:
        t, v = insn.operands[i]
        size = _operand_size(insn.name, insn.specs[i])
        if t == OPER_TYPE.GPR:
            params.append(_r(il, v))
        elif t in (OPER_TYPE.IMM, OPER_TYPE.UIMM):
            params.append(il.const(8, v))
        elif t == OPER_TYPE.ADDR:
            params.append(il.const_pointer(8, v))
        else:
            params.append(il.reg(size, _reg_for(insn, i, size)))
    il.append(il.intrinsic(outputs, intrinsic_name(insn.name), params))
