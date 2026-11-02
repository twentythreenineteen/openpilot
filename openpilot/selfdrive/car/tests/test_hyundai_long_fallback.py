"""A failed Hyundai LONG takeover must be visible in the cloud and in the car."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

from opendbc.car.hyundai.ecu_startup import Outcome
from openpilot.selfdrive.car import card as card_module
from openpilot.selfdrive.selfdrived import helpers


def test_stock_restored_emits_event_and_one_shot_flag():
  params = Mock()
  owner = SimpleNamespace(outcome=Outcome.STOCK_RESTORED)
  cp = SimpleNamespace(carFingerprint='HYUNDAI_IONIQ_5_N')

  with patch.object(card_module.cloudlog, 'event') as event:
    card_module.report_hyundai_long_fallback(owner, cp, params)

  event.assert_called_once_with('hyundai_long_fallback', car='HYUNDAI_IONIQ_5_N', owner='SimpleNamespace')
  params.put_bool.assert_called_once_with('HyundaiLongFallback', True, block=True)


def test_other_outcomes_stay_silent():
  params = Mock()
  for outcome in (None, Outcome.PENDING, Outcome.SENT_UNCONFIRMED, Outcome.STOCK_UNTOUCHED):
    with patch.object(card_module.cloudlog, 'event') as event:
      card_module.report_hyundai_long_fallback(SimpleNamespace(outcome=outcome), SimpleNamespace(carFingerprint='x'), params)
    event.assert_not_called()
  params.put_bool.assert_not_called()


def test_alert_fires_once_per_flag_then_expires():
  params = Mock()
  params.get_bool.return_value = True
  owner = SimpleNamespace(params=params, sm=SimpleNamespace())

  alert = helpers.hyundai_long_fallback_alert(owner, now=100.0)
  assert alert is not None
  assert alert.alert_type == 'hyundaiLongFallback/warning'
  assert alert.event_type == helpers.HYUNDAI_LONG_FALLBACK
  params.remove.assert_called_once_with('HyundaiLongFallback')

  # Flag consumed: the alert holds for its window, then clears without re-reading.
  params.get_bool.return_value = False
  assert helpers.hyundai_long_fallback_alert(owner, now=100.0 + helpers.HYUNDAI_LONG_FALLBACK_S - 0.1) is not None
  assert helpers.hyundai_long_fallback_alert(owner, now=100.0 + helpers.HYUNDAI_LONG_FALLBACK_S) is None
  assert params.remove.call_count == 1


def test_no_alert_without_a_failure():
  owner = SimpleNamespace(params=Mock(get_bool=Mock(return_value=False)), sm=SimpleNamespace())
  assert helpers.hyundai_long_fallback_alert(owner, now=100.0) is None
