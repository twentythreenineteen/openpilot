"""Reach test for the IONIQ 5 N CCNC-long safety word (bit 15, 0x8000).

The N's live word is 0x8895 = 0x0095 | HAS_LDA_BUTTON(0x0800) | CCNC(0x8000).
0x8895 already reaches all 5 CCNC frames and installs the AOL policy -- the
defect was never the word. canfd_stock_aol.qualified_long() compared the raw
word against word|0x804, where _base_word() returns 0x0091 and so expects
0x0895. The N's bit 15 -- the selector its own AOL policy adds -- made that
exact-equality test fail, so finalize_aol_configuration() raised and the car
could not start. The fix adds the CCNC bit to the expected word instead of
masking it, so only CCNC cars move and IONIQ 5 / Kona stay at 0x0895.

Everything here is resolved by executing the real hyundai_canfd_init() through
libsafety, or the real policy_for() through the real Python port. Nothing is
inferred from reading the source. libsafety is built with -DALLOW_DEBUG
(libsafety_py.py:31), which is required because hyundai_common_init() only
derives hyundai_longitudinal under ALLOW_DEBUG (modes/hyundai_common.h:118-123).

Measured deltas this file pins (real tree, BUILD_EXIT=0, RUN_EXIT=0):

  word     HEAD                  patched               aol_policy
  0x8895   tx=16  2/5  SET      tx=19  5/5  SET      SET   <- the fix
  0x8095   tx=16  2/5  NULL     tx=19  5/5  NULL     NULL  <- 6.7.9's word, DEAD
  0x0895   tx=13  0/5  SET      tx=13  0/5  SET      SET   <- IONIQ 5 / Kona, unchanged
  0x0C95   tx=0   0/5  NULL     tx=0   0/5  NULL     NULL  <- bit 10, POISONOUS
  0x0495   tx=13  0/5  NULL     tx=13  0/5  NULL     NULL  <- naive fix, DEAD

0x8895 does not extend 0x0895's table -- it selects a different one, so the
delta is +7 frames and -1 (0x1CF). The 0x0C95 / 0x0495 / 0x8095 cases are kept
as red-to-green tripwires: they are the three measured dead ends, and deleting
them lets the next agent rediscover them from source and retry.
"""
from opendbc.safety.tests.libsafety import libsafety_py
from opendbc.car import gen_empty_fingerprint
from opendbc.car.hyundai.interface import CarInterface
from opendbc.car.hyundai.values import CAR, HyundaiFlags
from openpilot.starpilot.car.hyundai.aol import policy_for, native_profile_supported

libsafety_py.ffi.cdef('extern safety_config current_safety_config;', override=True)
libsafety_py.ffi.cdef('extern const AolSafetyPolicy *aol_policy;', override=True)

SAFETY_HYUNDAI_CANFD = 28

# 0x895 is only admitted at the AOL runtime regime (alternative_experience == 32).
# At alt_exp 0 it resolves to an empty config, which would make every assertion
# about it vacuous.
AOL_ALT_EXP = 32

N_TODAY = 0x0895          # 0x0095 | HAS_LDA_BUTTON(0x0800) -- what the car runs now
N_FIXED = 0x8895          # N_TODAY | CAN_REFRESH_MSGS(0x8000) -- the fix
N_679_WORD = 0x8095       # 0x0095 | CAN_REFRESH_MSGS(0x8000) -- 6.7.9's literal word
N_BIT10 = 0x0C95          # N_TODAY | CCNC(0x0400) -- the retracted naive fix
N_NAIVE = 0x0495          # 0x0095 | CCNC(0x0400) -- the other retracted naive fix

# The five CCNC support frames. Post-fix all five are present for 0x8895.
CCNC_FRAMES = ((0x161, 1, 32), (0x162, 1, 32), (0x1BA, 1, 24), (0x1E5, 1, 16), (0x38C, 1, 32))

# The three frames the fix adds to ioniq6_long_lkas_alt_tx_msgs.
ADDED_FRAMES = ((0x100, 0, 24), (0x11A, 1, 16), (0x161, 1, 32), (0x162, 1, 32),
                (0x1BA, 1, 24), (0x1E5, 1, 16), (0x38C, 1, 32))
DROPPED_FRAMES = ((0x1CF, 1, 8),)


def resolve(param, alt_exp=AOL_ALT_EXP):
  """Run the production safety init for param and return (rc, config, tx set)."""
  safety = libsafety_py.libsafety
  safety.set_alternative_experience(alt_exp)
  rc = safety.set_safety_hooks(SAFETY_HYUNDAI_CANFD, param)
  config = safety.current_safety_config
  tx = frozenset((config.tx_msgs[i].addr, config.tx_msgs[i].bus, config.tx_msgs[i].len)
                 for i in range(config.tx_msgs_len))
  return rc, config, tx


def aol_policy_state():
  """Return 'SET' or 'NULL' for the global aol_policy (safety.h:61).

  Read after resolve(): set_safety_hooks() clears aol_policy to NULL (safety.h:508)
  before calling init(), so this reflects exactly what this param installed.
  """
  return 'SET' if libsafety_py.libsafety.aol_policy != libsafety_py.ffi.NULL else 'NULL'


def frames_present(tx):
  return sum(1 for f in CCNC_FRAMES if f in tx)


# ---------------------------------------------------------------------------
# Safety layer
# ---------------------------------------------------------------------------

def test_fixed_word_reaches_all_ccnc_frames_and_keeps_the_policy():
  """0x8895 must reach 5/5 frames AND install the AOL policy.

  FAILS ON UNMODIFIED HEAD: tx=16, 2/5 -- 0x161/0x162/0x38C are absent from
  ioniq6_long_lkas_alt_tx_msgs, so the policy is attached but three support
  frames are never sent.
  """
  rc, config, tx = resolve(N_FIXED)
  assert rc == 0
  assert frames_present(tx) == 5, f"0x8895 reached {frames_present(tx)}/5 CCNC frames, expected 5"
  for frame in CCNC_FRAMES:
    assert frame in tx, f"0x{frame[0]:03X} bus {frame[1]} missing from 0x8895 TX table"
  assert aol_policy_state() == 'SET', "0x8895 must install the AOL policy"
  assert config.tx_msgs_len == 19, f"0x8895 TX table is {config.tx_msgs_len} entries, expected 19"


def test_fixed_word_switches_to_the_alt_long_table():
  """0x8895 does not extend today's table -- it selects a different one.

  Measured: 0x0895 is HYUNDAI_CANFD_LKA_STEER_MSG_ALT_LONG_TX_MSGS (13 entries,
  0/5 CCNC frames); 0x8895 is ioniq6_long_lkas_alt_tx_msgs (19 entries, 5/5).
  So the delta is +7 frames and -1 (0x1CF, the cruise button, which the alt-long
  table replaces with the IONIQ 6 set). Pinning the exact delta guards against a
  table edit that silently drops coverage.
  """
  _, _, today_tx = resolve(N_TODAY)
  _, _, fixed_tx = resolve(N_FIXED)
  added = fixed_tx - today_tx
  dropped = today_tx - fixed_tx
  assert added == frozenset(ADDED_FRAMES), f"unexpected added frames: {sorted(added)}"
  assert dropped == frozenset(DROPPED_FRAMES), f"unexpected dropped frames: {sorted(dropped)}"


def test_today_word_is_unchanged():
  """0x0895 must stay byte-identical -- the fix must not touch the current path."""
  rc, config, tx = resolve(N_TODAY)
  assert rc == 0
  assert config.tx_msgs_len == 13
  assert frames_present(tx) == 0
  assert aol_policy_state() == 'SET'


def test_bit10_is_poisonous_and_bit11_is_required():
  """0x0C95 drops ALL TX -- bit 10 is not a spare on a bit-11 word.

  hyundai_canfd_stock_torque_aol_param() (hyundai_canfd_aol.h:5-19) requires
  !ccnc || topology in (8, 40). 0x0C95 strips bit 11, so base=0x0495,
  topology = base & 0x00B8 = 0x90 (144), ccnc = (base & 0x0400) != 0 -> true.
  Neither 8 nor 40, so the predicate is false, aol_policy is never attached,
  marker_valid goes false and ALL TX is dropped.

  The predicate is `static inline` and not cdef'd, so it cannot be called
  directly. The reason is asserted by contrast instead, which is stronger:
  bit 11 alone installs the policy, bit 10 alone kills it.
  """
  rc, config, tx = resolve(N_BIT10)
  assert rc == 0, "set_safety_hooks rejected 0x0C95 -- it is accepted, then inert"
  assert config.tx_msgs_len == 0, f"0x0C95 produced {config.tx_msgs_len} TX entries, expected 0"
  assert aol_policy_state() == 'NULL', "0x0C95 must NOT install the AOL policy"
  # Contrast: bit 11 on the same base installs the policy and sends frames.
  _, _, bit11_tx = resolve(N_TODAY)
  assert aol_policy_state() == 'SET'
  assert len(bit11_tx) == 13


def test_naive_ccnc_word_is_inert_not_poisonous():
  """0x0495 is inert: normal TX, but no policy and no CCNC frames."""
  rc, config, tx = resolve(N_NAIVE)
  assert rc == 0
  assert config.tx_msgs_len == 13
  assert frames_present(tx) == 0
  assert aol_policy_state() == 'NULL', "0x0495 must NOT install the AOL policy"


def test_679_word_sends_frames_but_has_no_policy():
  """0x8095 reaches 5/5 frames yet aol_policy is NULL -- 6.7.9's word is DEAD in 7.0.

  This is the trap that cost the most time: 6.7.9 really did transmit these
  frames, so "restore 6.7.9's word" looks correct and is not. The frames are
  necessary but not sufficient; the policy is what makes the axis work.
  """
  rc, config, tx = resolve(N_679_WORD)
  assert rc == 0
  assert frames_present(tx) == 5, "0x8095 should reach all 5 frames post-patch"
  assert aol_policy_state() == 'NULL', "0x8095 must NOT install the AOL policy -- it is dead"


def test_only_the_two_ccnc_words_moved():
  """Sweep: exactly 0x8895 and 0x8095 changed; every other word is untouched.

  0x8095 moves because it shares the bit-15 table branch. It gains frames but
  still gets no policy, so it stays dead.
  """
  words = (0x0895, 0x8815, 0x8015, 0x8895, 0x8095, 0x0C95, 0x0495, 0x8C95, 0x9095, 0x9015)
  lens = {}
  for w in words:
    _, config, _ = resolve(w)
    lens[w] = config.tx_msgs_len
  assert lens[0x8895] == 19
  assert lens[0x8095] == 19
  assert lens[0x0895] == 13
  assert lens[0x8815] == 16
  assert lens[0x8015] == 16
  for w in (0x0C95, 0x8C95, 0x9095, 0x9015):
    assert lens[w] == 0, f"0x{w:04X} should produce no TX, got {lens[w]}"


# ---------------------------------------------------------------------------
# Python layer -- the word the port actually hands to safety
# ---------------------------------------------------------------------------

def _fp():
  """A fingerprint that makes the N resolve to LKA-steer + CCNC + long.

  0x110 on bus 2 is the signal that sets lka_steering (interface.py:141), which
  is what makes the N take the qualified_canfd_long branch rather than stock.
  """
  fp = gen_empty_fingerprint()
  fp[2][0x110] = 32
  fp[2][0x362] = 32
  fp[1].update({0x35: 32, 0xa0: 24, 0xea: 24, 0x130: 32, 0x175: 24, 0x1aa: 16,
                0x1ba: 32, 0x1cf: 8, 0x1e5: 32, 0x36a: 32})
  return fp


def _cp(fingerprint):
  """Build a real CP through the real interface, as card.py does."""
  return CarInterface.get_params(fingerprint, _fp(), [], True, False, False)


def _candidate(fingerprint):
  """The CP the AOL policy is actually computed on, as card.py:376 does.

  policy_for() must see the candidate, not the raw interface CP: the
  qualified_canfd_long branch that carries the CCNC bit-15 addition is only
  reachable once the stock CP has been converted to the LONG candidate.
  """
  from opendbc.car.hyundai import torque_ev_startup as tes
  return tes.candidate(_cp(fingerprint), requested=True, is_release=False)


def test_n_gets_bit15_and_other_ccnc_cars_do_not():
  """The N's policy addition must be 0x8800; the IONIQ 5 and Kona stay 0x0800.

  FAILS ON UNMODIFIED HEAD: every CCNC-long car gets 0x0800, so the N's word
  stays 0x0895 and never reaches the bit-15 table branch.
  """
  n_cp = _cp(CAR.HYUNDAI_IONIQ_5_N)
  assert n_cp.flags & HyundaiFlags.CCNC, "the N must carry CCNC or this test is vacuous"
  assert n_cp.flags & HyundaiFlags.CANFD_LKA_STEER_MSG, "the N must be LKA-steer"
  n_policy = policy_for(_candidate(CAR.HYUNDAI_IONIQ_5_N))
  assert n_policy.safety_param_addition == 0x8800, \
    f"N addition is 0x{n_policy.safety_param_addition:04X}, expected 0x8800"

  for fp in (CAR.HYUNDAI_IONIQ_5, CAR.HYUNDAI_KONA_EV_2ND_GEN):
    cp = _cp(fp)
    assert not cp.flags & HyundaiFlags.CCNC, f"{fp} must not carry CCNC"
    assert policy_for(_candidate(fp)).safety_param_addition == 0x0800, f"{fp} must stay 0x0800"


def test_n_final_word_is_the_one_that_works():
  """Compose the port's addition onto the base word and check it lands on 0x8895.

  This is card.py:402's arithmetic: the candidate word 0x0095, then |= the
  policy's addition.
  """
  policy = policy_for(_candidate(CAR.HYUNDAI_IONIQ_5_N))
  final = 0x0095 | policy.safety_param_addition
  assert final == N_FIXED, f"composed word is 0x{final:04X}, expected 0x8895"


def test_both_words_are_admitted_native_profiles():
  """0x8895 is a pre-existing recognised profile, not a word the patch invents.

  IONIQ6_AOL_SAFETY_PARAMS (aol.py:42) already contains 0x8895, so the fix
  steers the N onto a word the AOL runtime already knows and admits.
  """
  assert native_profile_supported(SAFETY_HYUNDAI_CANFD, N_FIXED) is True
  assert native_profile_supported(SAFETY_HYUNDAI_CANFD, N_TODAY) is True


def test_dead_words_are_rejected_by_the_runtime_admission_gate():
  """0x8095 / 0x0C95 / 0x0495 are refused by a second, independent subsystem.

  The safety policy already returns NULL for these. native_profile_supported
  refuses them too -- two subsystems, same verdict.
  """
  for w in (N_679_WORD, N_BIT10, N_NAIVE):
    assert native_profile_supported(SAFETY_HYUNDAI_CANFD, w) is False, \
      f"0x{w:04X} should be refused by native_profile_supported"
