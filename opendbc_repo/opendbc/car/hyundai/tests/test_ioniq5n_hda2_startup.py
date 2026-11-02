"""Unit and integration tests for Hyundai IONIQ 5 N ccNC HDA-II startup and lateral configuration."""
from copy import deepcopy
import time
from unittest.mock import MagicMock, patch

from opendbc.car import gen_empty_fingerprint, structs
from opendbc.car.carlog import carlog
from opendbc.car.hyundai.canfd_stock_aol import qualified as stock_qualified
from opendbc.car.hyundai.ecu_startup import Outcome
from opendbc.car.hyundai.generic_hda2_startup import (
  CARS,
  GenericHDA2Startup,
  qualified,
  required,
)
from opendbc.car.hyundai.interface import CarInterface
def params(identity, hybrid=False, alpha=True, alt=False):
  fp = gen_empty_fingerprint()
  fp[1].update({0x1CF: 8, 0x130: 32, 0x35: 32, 0x175: 24, 0xA0: 24, 0xEA: 24, 0x1A0: 32})
  if hybrid:
    fp[1][0xFA] = 8
  fp[2].update({0x110: 32, 0x362: 32} if alt else {0x50: 16, 0x2A4: 24})
  fw = [structs.CarParams.CarFw(ecu=structs.CarParams.Ecu.adas, address=0x730, bus=1)]
  return CarInterface.get_params(identity, fp, fw, alpha, False, False)
from opendbc.car.hyundai.values import CAR, DBC, Bus, HYUNDAI_MRR35_RADAR_DBC, CarControllerParams


def make_ioniq5n_stock_cp(alpha=False):
  return params(CAR.HYUNDAI_IONIQ_5_N, hybrid=False, alpha=alpha, alt=False)


def make_ioniq5n_long_cp():
  return params(CAR.HYUNDAI_IONIQ_5_N, hybrid=False, alpha=True, alt=False)


class MockKeeper:
  def __init__(self, send_fn, clock=time.monotonic):
    self.send_fn = send_fn
    self.started = False
    self.stopped = False
    self.abort_reason = None

  def start(self):
    self.started = True

  def stop(self):
    self.stopped = True


def test_ioniq5n_specs_and_lateral_parameters():
  stock_cp = make_ioniq5n_stock_cp()
  assert stock_cp.carFingerprint == CAR.HYUNDAI_IONIQ_5_N
  # Restored factory steer ratio
  assert abs(stock_cp.steerRatio - 14.26) < 1e-4
  # MRR35 radar DBC
  assert DBC[stock_cp.carFingerprint][Bus.radar] == HYUNDAI_MRR35_RADAR_DBC

  # Standard responsive slew rate (10 up / 8 down) under 15 m/s, NOT asymmetric choked 3/7
  params_low = CarControllerParams(stock_cp, 10.0)
  assert params_low.STEER_DELTA_UP == 10
  assert params_low.STEER_DELTA_DOWN == 8


from opendbc.car.hyundai.torque_ev_startup import TorqueEVStartup, candidate

def test_ioniq5n_qualifies_for_torque_ev_startup():
  stock_cp = make_ioniq5n_stock_cp(alpha=False)
  owner = CarInterface.startup_owner(stock_cp, (list, list), requested=True)
  assert isinstance(owner, TorqueEVStartup)
  assert owner.communication_control_request() == b"\x28\x83\x01"


def test_prepare_when_not_admitted_safely_falls_back():
  """When admission is false (e.g. ignition off or controls not admitted),
  TorqueEVStartup cleanly preserves stock SCC without sending UDS commands.
  """
  stock_cp = make_ioniq5n_stock_cp(alpha=False)
  sent_frames = []

  startup = CarInterface.startup_owner(stock_cp, (list, sent_frames.extend), requested=True)
  assert startup.communication_control_request() == b"\x28\x83\x01"

  prepared = startup.prepare(admission=lambda: False)
  assert startup.outcome is Outcome.STOCK_UNTOUCHED
  assert not prepared.openpilotLongitudinalControl
  assert prepared.pcmCruise
  assert len(sent_frames) == 0


def test_prepare_in_acc_mode_takes_over_longitudinal():
  """When vehicle is in ACC mode (not READY), prepare sends 0x28 83 01 and takes over longitudinal."""
  stock_cp = make_ioniq5n_stock_cp(alpha=False)
  sent_frames = []

  startup = CarInterface.startup_owner(stock_cp, (list, sent_frames.extend), requested=True)

  # Mock successful UDS exchange (0x10 03 positive -> 0x28 83 01 suppressed silence)
  def mock_query(sender, receiver, bus, addresses, commands, responses):
    cmd = commands[0]
    sender([(0x730, bytes([len(cmd)]) + cmd + bytes(7 - len(cmd)), bus)])
    mock_resp = MagicMock()
    if cmd == b"\x10\x03":
      mock_resp.get_data.return_value = {(0x730, None): b"\x50\x03"}
    elif cmd == b"\x28\x83\x01":
      mock_resp.get_data.return_value = {}
    else:
      mock_resp.get_data.return_value = {(0x730, None): b""}
    return mock_resp

  startup._query = mock_query

  prepared = startup.prepare(admission=lambda: True)
  assert startup.outcome is Outcome.SENT_UNCONFIRMED
  assert startup.keeper is not None
  assert prepared.openpilotLongitudinalControl


def test_prepare_ccnc_nrc22_clean_fallback_no_crash():
  """When ccNC gateway rejects takeover with NRC 0x22 (or car is in READY mode),
  prepare MUST NOT raise RuntimeError.
  It must smoothly fall back to STOCK_RESTORED and strip LONG from prepared CP.
  """
  stock_cp = make_ioniq5n_stock_cp(alpha=False)
  sent_frames = []

  startup = CarInterface.startup_owner(stock_cp, (list, sent_frames.extend), requested=True)

  # Mock UDS exchange returning NRC 0x22 (Conditions Not Correct: 7F 28 22)
  def mock_query(sender, receiver, bus, addresses, commands, responses):
    cmd = commands[0]
    sender([(0x730, bytes([len(cmd)]) + cmd + bytes(7 - len(cmd)), bus)])
    mock_resp = MagicMock()
    if cmd == b"\x10\x03":
      mock_resp.get_data.return_value = {(0x730, None): b"\x50\x03"}
    elif cmd == b"\x28\x83\x01":
      mock_resp.get_data.return_value = {(0x730, None): b"\x7f\x28\x22"}
    elif cmd == b"\x10\x01":
      mock_resp.get_data.return_value = {(0x730, None): b"\x50\x01"}
    else:
      mock_resp.get_data.return_value = {(0x730, None): b""}
    return mock_resp

  startup._query = mock_query

  # MUST NOT RAISE RuntimeError or NameError!
  prepared = startup.prepare(admission=lambda: True)
  assert startup.outcome is Outcome.STOCK_RESTORED
  assert not prepared.openpilotLongitudinalControl
  assert prepared.pcmCruise


def test_ccnc_fallback_is_logged():
  """The STOCK_RESTORED fallback must not be silent."""
  stock_cp = make_ioniq5n_stock_cp(alpha=False)
  sent_frames = []
  startup = CarInterface.startup_owner(stock_cp, (list, sent_frames.extend), requested=True)

  def mock_query(sender, receiver, bus, addresses, commands, responses):
    cmd = commands[0]
    sender([(0x730, bytes([len(cmd)]) + cmd + bytes(7 - len(cmd)), bus)])
    mock_resp = MagicMock()
    if cmd == b"\x10\x03":
      mock_resp.get_data.return_value = {(0x730, None): b"\x50\x03"}
    elif cmd == b"\x28\x83\x01":
      mock_resp.get_data.return_value = {(0x730, None): b"\x7f\x28\x22"}
    elif cmd == b"\x10\x01":
      mock_resp.get_data.return_value = {(0x730, None): b"\x50\x01"}
    else:
      mock_resp.get_data.return_value = {(0x730, None): b""}
    return mock_resp

  startup._query = mock_query

  with patch.object(carlog, 'warning') as logged:
    startup.prepare(admission=lambda: True)

  assert startup.outcome is Outcome.STOCK_RESTORED
  assert logged.call_count >= 1
  assert any('retaining stock longitudinal' in call[0][0] for call in logged.call_args_list)

