"""Regression test for the finalize hop -- the one the earlier suite missed.

finalize_aol_configuration() builds its own expected CarParams and compares them
to the card's. It hardcoded `|= 0x0800`, so for a CCNC car it expected 0x0895
while the card builds 0x8895, and raised "changed unrelated CarParams". The
safety layer and the admission gate were both already correct; this was a third,
independent site with the same asymmetry.

This test drives the real startup object through the real card order and asserts
finalize RETURNS for the N. It fails on the pre-fix tree with that exact message.
"""
from opendbc.car import gen_empty_fingerprint
from opendbc.car.hyundai.interface import CarInterface
from opendbc.car.hyundai.values import CAR
from opendbc.car.hyundai import torque_ev_startup as tes
from openpilot.starpilot.car.hyundai.aol import policy_for, aol_safety_config


def _fp():
  fp = gen_empty_fingerprint()
  fp[2][0x110] = 32
  fp[2][0x362] = 32
  fp[1].update({0x35: 32, 0xa0: 24, 0xea: 24, 0x130: 32, 0x175: 24, 0x1aa: 16,
                0x1ba: 32, 0x1cf: 8, 0x1e5: 32, 0x36a: 32})
  return fp


class _FakeCI:
  def __init__(self, cp):
    self.CP = cp


def _run(car):
  """Drive card.py's real order and return (word, finalize outcome)."""
  stock = CarInterface.get_params(car, _fp(), [], True, False, False)
  cand = tes.candidate(stock, requested=True, is_release=False)
  assert cand is not None, f"{car} is not a torque-EV long candidate"

  obj = tes.TorqueEVStartup(cand, [None, lambda *a, **k: None], stock_cp=stock)
  obj.prepare_when_controls_ready(admission=lambda: True)

  # card.py:399-403 -- the card mutates the candidate config in place.
  policy = policy_for(cand)
  config = aol_safety_config(cand)
  config.safetyParam |= policy.safety_param_addition
  cand.alternativeExperience |= policy.alternative_experience_addition

  obj.ci = _FakeCI(cand)
  obj.ready = True
  obj.finalize_aol_configuration(obj.ci)
  return int(config.safetyParam)


def test_finalize_returns_for_the_n():
  """The N must complete the startup contract, not just clear the admission gate.

  FAILS ON PRE-FIX: RuntimeError('Torque EV AOL finalization changed unrelated
  CarParams') -- finalize expected 0x0895, the card built 0x8895.
  """
  word = _run(CAR.HYUNDAI_IONIQ_5_N)
  assert word == 0x8895, f"N final word is 0x{word:04X}, expected 0x8895"


def test_finalize_returns_for_the_non_ccnc_controls():
  """IONIQ 5 and Kona EV 2nd Gen must keep finalizing at 0x0895, unchanged."""
  for car in (CAR.HYUNDAI_IONIQ_5, CAR.HYUNDAI_KONA_EV_2ND_GEN):
    word = _run(car)
    assert word == 0x0895, f"{car} final word is 0x{word:04X}, expected 0x0895"
