import struct

from binaryninja import Symbol
from binaryninja.enums import SymbolType

from . import ladis

R_LARCH_JUMP_SLOT = 5


def _cstr(bv, addr, limit=256):
    raw = bv.read(addr, limit) or b''
    return raw.split(b'\0', 1)[0].decode('utf-8', 'replace') #text


def _stub_got_slot(bv, addr):
    a = ladis.decode(bv.read(addr, 4), addr)
    b = ladis.decode(bv.read(addr + 4, 4), addr + 4)
    if a is None or b is None or a.name != 'pcaddu12i' or b.name != 'ld.d':
        return None
    reg, hi = a.operands[0][1], a.operands[1][1]
    if b.operands[1][1] != reg:
        return None
    return (addr + (hi << 12) + b.operands[2][1]) & 0xffffffffffffffff


def name_plt_stubs(bv):
    plt = bv.get_section_by_name('.plt')
    rela = bv.get_section_by_name('.rela.plt')
    dynsym = bv.get_section_by_name('.dynsym')
    dynstr = bv.get_section_by_name('.dynstr')
    if not (plt and rela and dynsym and dynstr):
        return 0

    got_names = {}
    data = bv.read(rela.start, rela.end - rela.start) or b''
    for off in range(0, len(data) - 23, 24):
        r_offset, r_info, _ = struct.unpack_from('<QQq', data, off)
        if r_info & 0xffffffff != R_LARCH_JUMP_SLOT:
            continue
        sym = bv.read(dynsym.start + 24 * (r_info >> 32), 24)
        if not sym or len(sym) < 4:
            continue
        name = _cstr(bv, dynstr.start + struct.unpack_from('<I', sym)[0])
        if name:
            got_names[r_offset] = name

    count = 0
    for addr in range(plt.start, plt.end - 15, 4):
        slot = _stub_got_slot(bv, addr)
        name = got_names.get(slot) if slot is not None else None
        if name is None:
            continue
        bv.define_auto_symbol(Symbol(SymbolType.ImportAddressSymbol, slot, name))
        bv.define_auto_symbol(Symbol(SymbolType.ImportedFunctionSymbol, addr, name))
        bv.add_function(addr)
        count += 1
    return count
