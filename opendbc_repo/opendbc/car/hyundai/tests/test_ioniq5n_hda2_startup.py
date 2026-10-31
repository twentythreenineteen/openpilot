"""Unit and integration tests for Hyundai IONIQ 5 N ccNC HDA-II startup and lateral configuration."""
from copy import deepcopy
import time
from unittest.mock import MagicMock

import pytest

from opendbc.car import gen_empty_fingerprint, structs
from opendbc.car.hyundai.canfd_stock_aol import qualified as stock_qualified
from opendbc.car.hyundai.ecu_startup import Outcome
from opendbc.car.hyundai.generic_hda2_startup import (
  CARS,
  GenericHDA2Startup,
  qualified,
  required,
)
from opendbc.car.hyundai.interface import CarInterface
from opendbc.car.hyundai.tests.test_radar_hda2_startup import params
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


def test_ioniq5n_qualifies_for_generic_hda2_startup():
  stock_cp = make_ioniq5n_stock_cp(alpha=False)
  assert stock_cp.carFingerprint in CARS
  assert not required(stock_cp)

  long_cp = make_ioniq5n_long_cp()
  assert required(long_cp)
  assert qualified(long_cp)

  owner = CarInterface.startup_owner(long_cp, (list, list), requested=True)
  assert isinstance(owner, GenericHDA2Startup)
  assert owner.communication_control_request() == b"\x28\x83\x01"


def test_prepare_when_not_admitted_safely_falls_back():
  """When admission is false (e.g. ignition off or controls not admitted),

  GenericHDA2Startup cleanly preserves stock SCC without sending UDS commands.
  """
  long_cp = make_ioniq5n_long_cp()
  sent_frames = []

  startup = GenericHDA2Startup(long_cp, (list, sent_frames.extend), keeper_factory=MockKeeper)
  assert startup.communication_control_request() == b"\x28\x83\x01"

  prepared = startup.prepare(admission=lambda: False)
  assert startup.outcome is Outcome.STOCK_UNTOUCHED
  assert not prepared.openpilotLongitudinalControl
  assert prepared.pcmCruise
  assert prepared.safetyConfigs[0].safetyParam == 0x11
  assert len(sent_frames) == 0


def test_prepare_in_acc_mode_takes_over_longitudinal():
  """When vehicle is in ACC mode (not READY), prepare sends 0x28 83 01 and takes over longitudinal."""
  long_cp = make_ioniq5n_long_cp()
  sent_frames = []

  startup = GenericHDA2Startup(long_cp, (list, sent_frames.extend), keeper_factory=MockKeeper)

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
  assert startup.keeper is not None and startup.keeper.started
  assert prepared.openpilotLongitudinalControl


def test_prepare_ccnc_nrc22_clean_fallback_no_crash():
  """When ccNC gateway rejects takeover with NRC 0x22 (or car is in READY mode),

  prepare MUST NOT raise RuntimeError.
  It must smoothly fall back to STOCK_RESTORED and strip LONG from prepared CP.
  """
  long_cp = make_ioniq5n_long_cp()
  sent_frames = []

  startup = GenericHDA2Startup(long_cp, (list, sent_frames.extend), keeper_factory=MockKeeper)

  # Mock UDS exchange returning NRC 0x22 (Conditions Not Correct: 7F 28 22)
  # and restore also rejected because car is in READY mode
  def mock_query(sender, receiver, bus, addresses, commands, responses):
    cmd = commands[0]
    sender([(0x730, bytes([len(cmd)]) + cmd + bytes(7 - len(cmd)), bus)])
    mock_resp = MagicMock()
    if cmd == b"\x10\x03":
      mock_resp.get_data.return_value = {(0x730, None): b"\x50\x03"}
    elif cmd == b"\x28\x83\x01":
      # Rejection from gateway
      mock_resp.get_data.return_value = {(0x730, None): b"\x7f\x28\x22"}
    else:
      mock_resp.get_data.return_value = {(0x730, None): b"\x7f\x28\x22"}
    return mock_resp

  startup._query = mock_query

  # MUST NOT RAISE RuntimeError!
  prepared = startup.prepare(admission=lambda: True)
  assert startup.outcome is Outcome.STOCK_RESTORED
  assert not prepared.openpilotLongitudinalControl
  assert prepared.pcmCruise
  assert prepared.safetyConfigs[0].safetyParam == 0x11

  startup._query = mock_query

  # MUST NOT RAISE RuntimeError!
  prepared = startup.prepare(admission=lambda: True)
  assert startup.outcome is Outcome.STOCK_RESTORED
  assert not prepared.openpilotLongitudinalControl
  assert prepared.pcmCruise
  assert prepared.safetyConfigs[0].safetyParam == 0x11
