"""Element symbols H..Rn and atomic numbers."""

from __future__ import annotations

_SYMBOL_TEXT = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
    "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn"
)
_SYMBOLS = tuple(_SYMBOL_TEXT.split())

_Z_BY_SYMBOL = {symbol: z for z, symbol in enumerate(_SYMBOLS, start=1)}


def normalize_symbol(symbol: str) -> str:
    candidate = symbol.strip().capitalize()
    if candidate not in _Z_BY_SYMBOL:
        raise ValueError(f"Unknown element symbol {candidate!r}")
    return candidate


def atomic_number(symbol: str) -> int:
    return _Z_BY_SYMBOL[normalize_symbol(symbol)]
