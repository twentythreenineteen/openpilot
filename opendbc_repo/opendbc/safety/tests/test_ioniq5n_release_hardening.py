"""Release-build reach test for the IONIQ 5 N CCNC-long word (0x8895).

The N's live panda is a DEBUG build, so the safety layer runs with ALLOW_DEBUG
defined and the word works. A release build (ALLOW_DEBUG undefined, as panda's
SConscript produces when RELEASE is set) previously rejected the N outright:
`valid = !GET_FLAG(param, 32768U)` dropped every bit-15 word before init ran.

The hardening keeps the release rejection for every bit-15 word EXCEPT the N's
exact word. This file pins that with a release-compiled libsafety
(_build_libsafety(release=True)), not by reading the preprocessor.

Measured on the hardened tree (release build, BUILD_EXIT=0):

  word      before hardening   after hardening
  0x8895    0                  19   <- the fix
  0x8095    0                  0    <- 6.7.9's word stays rejected
  0x8815    0                  0    <- other bit-15 words stay rejected
  0x8015    0                  0
  0x0895    0                  0    <- non-CCNC AOL word, unchanged
  0x0800    4                  4    <- stock words unchanged
  0x0010    3                  3
  0x0495    3                  3

FAILS ON UNMODIFIED HEAD: the first test sees frozenset() instead of N_TX.
"""
from functools import cache

from opendbc.safety.tests.libsafety import libsafety_py

libsafety_py.ffi.cdef('extern safety_config current_safety_config;', override=True)

SAFETY_HYUNDAI_CANFD = 28
AOL_ALT_EXP = 32

N_FIXED = 0x8895

# Exact release TX inventory for the N. Derived by dumping the hardened release
# build; 19 entries = 18 table entries + 0x11A appended by hyundai_daw_configure
# (modes/hyundai_daw.h:103-113) because daw_profile is true for 0x8895.
N_TX = frozenset({
  (0x110, 0, 32), (0x362, 0, 32), (0x12A, 1, 16), (0x1E0, 1, 16), (0x1A0, 1, 32),
  (0x100, 0, 24), (0x1BA, 1, 24), (0x1E5, 1, 16), (0x051, 0, 32), (0x730, 1, 8),
  (0x160, 1, 16), (0x1EA, 1, 32), (0x200, 1, 8), (0x345, 1, 8), (0x1DA, 1, 32),
  (0x161, 1, 32), (0x162, 1, 32), (0x38C, 1, 32), (0x11A, 1, 16),
})

# Stock words must be byte-identical between debug and release.
STOCK = {
  0x0800: frozenset({(0x1CF, 2, 8), (0x12A, 0, 16), (0x1E0, 0, 16), (0x1A0, 0, 32)}),
  0x0010: frozenset({(0x1CF, 1, 8), (0x050, 0, 16), (0x2A4, 0, 24)}),
}


@cache
def release_safety():
  """A second libsafety built without -DALLOW_DEBUG, leaving the default intact."""
  default = libsafety_py.libsafety
  library = libsafety_py.ffi.dlopen(libsafety_py._build_libsafety(release=True))
  assert libsafety_py.libsafety is default, "release build must not replace the module default"
  return library


def resolve(param, alt_exp=AOL_ALT_EXP):
  """Run the production safety init on the release library; return the TX set."""
  safety = release_safety()
  safety.set_alternative_experience(alt_exp)
  assert safety.set_safety_hooks(SAFETY_HYUNDAI_CANFD, param) == 0
  config = safety.current_safety_config
  return frozenset((config.tx_msgs[i].addr, config.tx_msgs[i].bus, config.tx_msgs[i].len)
                   for i in range(config.tx_msgs_len))


def test_release_admits_the_n_word_with_its_full_inventory():
  """0x8895 must reach its 19-entry table on release, not be dropped by the bit-15 gate."""
  assert resolve(N_FIXED) == N_TX


def test_release_still_rejects_every_other_bit15_word():
  """The hardening is exact-word: no other bit-15 word may reach a long table."""
  for word in (0x8095, 0x8815, 0x8015):
    assert resolve(word) == frozenset(), f"0x{word:04X} reached a TX table on release"


def test_release_stock_words_are_unchanged():
  """Narrowing the release predicate must not move any stock topology."""
  for word, expected in STOCK.items():
    assert resolve(word, alt_exp=0) == expected, f"0x{word:04X} moved on release"
