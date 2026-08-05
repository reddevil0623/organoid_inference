"""Shared path configuration. Edit these for your machine."""
from pathlib import Path
import sys

SCRIPTS    = Path(__file__).resolve().parent
PROJECT    = SCRIPTS.parent

MODEL_REPO = PROJECT / "MCPFM_tauV-model"

DATA       = PROJECT / "data" / "DATA"
DATA_SYNTH = PROJECT / "data" / "DATA_SYNTH"
TESTS_SYNTH= PROJECT / "data" / "TESTS_SYNTH"
TESTS_EXP  = PROJECT / "data" / "experimental_images"

OUTPUTS       = PROJECT / "outputs"
ABC_SMC       = OUTPUTS / "ABC_SMC"
OUT_SYNTH_012 = OUTPUTS / "OUT_SYNTH_012"

LIB = PROJECT / "lib"


def ensure_eucalc():
    """Add lib/ to sys.path so `import eucalc` works."""
    lib_str = str(LIB)
    if lib_str not in sys.path:
        sys.path.insert(0, lib_str)


if __name__ == "__main__":
    for name in ("PROJECT", "MODEL_REPO", "DATA", "DATA_SYNTH", "TESTS_SYNTH",
                 "TESTS_EXP", "OUTPUTS", "ABC_SMC", "OUT_SYNTH_012", "LIB"):
        p = globals()[name]
        exists = "OK" if p.exists() else "MISSING"
        print(f"  {name:20s} = {p}  [{exists}]")
