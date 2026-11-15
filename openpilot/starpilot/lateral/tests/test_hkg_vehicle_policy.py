import pytest

from opendbc.car.car_helpers import interfaces
from opendbc.car.hyundai.values import CAR
from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque
from openpilot.starpilot.lateral.controller_selection import ControllerMode, default_selection, replace_mode, selection_from_bytes
from openpilot.starpilot.lateral.hkg_vehicle_policy import HKGVehicleTorquePolicy, PROFILES, policy_for


CASES = [(vehicle, profile, factor) for profile, (vehicles, factor) in PROFILES.items() for vehicle in sorted(vehicles, key=str)]


@pytest.mark.parametrize('vehicle,profile,factor', CASES)
def test_exact_profile_defaults_and_saved_controller_selection(vehicle, profile, factor):
  cp = interfaces[vehicle].get_non_essential_params(vehicle)
  assert policy_for(cp) == profile
  expected = ControllerMode.STARPILOT
  assert default_selection(cp).mode == expected
  for raw in (b'{', b'{"version":true,"vehicles":{}}', b'{"version":1,"vehicles":{},"vehicles":{}}', b'x' * 4097):
    assert selection_from_bytes(cp, raw).mode == expected
  unrelated = interfaces[CAR.HYUNDAI_IONIQ_6].get_non_essential_params(CAR.HYUNDAI_IONIQ_6)
  assert selection_from_bytes(cp, replace_mode(None, unrelated, ControllerMode.STANDARD)).mode == expected
  saved = replace_mode(None, cp, ControllerMode.STARPILOT)
  choice = selection_from_bytes(cp, saved)
  assert choice.mode == ControllerMode.STARPILOT
  assert choice.policy == profile
  ci = interfaces[vehicle](cp)
  default = LatControlTorque(cp.as_reader(), ci, 0.01)
  assert bool(default.starpilot_extension) == (expected == ControllerMode.STARPILOT)
  standard_choice = selection_from_bytes(cp, replace_mode(saved, cp, ControllerMode.STANDARD))
  standard = LatControlTorque(cp.as_reader(), ci, 0.01, controller_mode=standard_choice.mode)
  assert standard.starpilot_extension is None
  assert standard.torque_params.latAccelFactor == pytest.approx(cp.lateralTuning.torque.latAccelFactor)
  controller = LatControlTorque(cp.as_reader(), ci, 0.01, controller_mode=choice.mode)
  policy = controller.starpilot_extension.policy
  assert isinstance(policy, HKGVehicleTorquePolicy)
  assert policy.profile == profile
  assert controller.torque_params.latAccelFactor == pytest.approx(cp.lateralTuning.torque.latAccelFactor * factor)
  assert controller.pid.pos_limit == pytest.approx(controller.lateral_accel_from_torque(controller.steer_max, cp.lateralTuning.torque))
  for _ in range(3):
    controller.update_torque_parameters(3.0, 0.02, 0.12)
    assert controller.torque_params.latAccelFactor == pytest.approx(3.0 * factor)
    assert controller.torque_params.latAccelOffset == pytest.approx(0.02)
    assert controller.torque_params.friction == pytest.approx(0.12)
  assert selection_from_bytes(cp, replace_mode(saved, cp, ControllerMode.STANDARD)).mode == ControllerMode.STANDARD


@pytest.mark.parametrize('vehicle,profile,factor', CASES)
def test_unadmitted_and_cross_vehicle_choices_are_not_applied(vehicle, profile, factor):
  cp = interfaces[vehicle].get_non_essential_params(vehicle)
  for field in ('passive', 'dashcamOnly', 'notCar'):
    other = cp.as_reader().as_builder()
    setattr(other, field, True)
    assert policy_for(other) is None
  cp.brand = 'gm'
  assert policy_for(cp) is None
  cp.brand = 'hyundai'
  saved = replace_mode(None, cp, ControllerMode.STARPILOT)
  other = interfaces[CAR.HYUNDAI_IONIQ_5].get_non_essential_params(CAR.HYUNDAI_IONIQ_5)
  if vehicle != CAR.HYUNDAI_IONIQ_5:
    assert selection_from_bytes(other, saved).mode == ControllerMode.STARPILOT
    assert selection_from_bytes(other, saved).source == 'default'


@pytest.mark.parametrize('vehicle,profile,factor', CASES)
def test_invalid_torque_contract_cannot_select_vehicle_policy(vehicle, profile, factor):
  cp = interfaces[vehicle].get_non_essential_params(vehicle)
  for field, value in (('latAccelFactor', 0.0), ('latAccelFactor', float('nan')), ('latAccelOffset', float('inf')), ('friction', -1.0)):
    other = cp.as_reader().as_builder()
    setattr(other.lateralTuning.torque, field, value)
    assert policy_for(other) is None
    assert selection_from_bytes(other, None).mode == ControllerMode.STANDARD
  cp.lateralTuning.init('pid')
  assert policy_for(cp) is None


@pytest.mark.parametrize('kph,desired,increment', (
  (60., 0., 0.), (100., .65, 0.), (110., .65, .30), (120., .65, .30),
  (120., -.65, .30), (120., 1.6, .15), (120., -1.6, .15),
  (120., 2.0, 0.), (140., 0., .30), (140., -2., 0.),
))
def test_kona_2022_high_speed_friction_boundaries(monkeypatch, kph, desired, increment):
  from openpilot.starpilot.lateral import hkg_vehicle_shaping as shaping
  threshold = shaping.get_kona_ev_2022_friction_threshold(kph / 3.6, desired)
  monkeypatch.setattr(shaping, 'KONA_EV_2022_HIGH_SPEED_FRICTION_THRESHOLD_GAIN', 0.)
  baseline = shaping.get_kona_ev_2022_friction_threshold(kph / 3.6, desired)
  assert threshold == pytest.approx(baseline + increment)
  assert baseline <= threshold <= baseline + .30


def test_kona_2022_high_speed_friction_reaches_selected_controller(monkeypatch):
  from types import SimpleNamespace
  from openpilot.cereal import log
  from opendbc.car.vehicle_model import VehicleModel
  from opendbc.car.lateral import get_friction
  from openpilot.starpilot.lateral import hkg_vehicle_shaping as shaping

  cp = interfaces[CAR.HYUNDAI_KONA_EV_2022].get_non_essential_params(CAR.HYUNDAI_KONA_EV_2022)
  cs = SimpleNamespace(vEgo=120. / 3.6, steeringAngleDeg=0., steeringPressed=False)
  params = log.VehicleParameters.new_message(angleOffsetDeg=0., roll=0.)

  def run():
    controller = LatControlTorque(cp.as_reader(), interfaces[cp.carFingerprint](cp), .01, controller_mode=ControllerMode.STARPILOT)
    for _ in range(60):
      output, _, state = controller.update(True, cs, VehicleModel(cp), params, False, .0001, False, .28)
    return controller, output, state

  controller, output, state = run()
  threshold = shaping.get_kona_ev_2022_friction_threshold(cs.vEgo, .65)
  standard = shaping.get_standard_friction_threshold(cs.vEgo)
  for direction in (-1., 1.):
    small = get_friction(direction * .1, 0., threshold, controller.torque_params)
    assert abs(small) < abs(get_friction(direction * .1, 0., standard, controller.torque_params))
    assert small * direction > 0.
    assert get_friction(direction, 0., threshold, controller.torque_params) == pytest.approx(
      get_friction(direction, 0., standard, controller.torque_params))
  monkeypatch.setattr(shaping, 'KONA_EV_2022_HIGH_SPEED_FRICTION_THRESHOLD_GAIN', 0.)
  baseline, base_output, base_state = run()
  assert state.active and base_state.active
  assert state.desiredLateralAccel == pytest.approx(base_state.desiredLateralAccel)
  assert state.p == pytest.approx(base_state.p)
  assert abs(state.f) < abs(base_state.f)
  assert abs(output) < abs(base_output)
  assert controller.steer_max == baseline.steer_max


@pytest.mark.parametrize('vehicle', (CAR.HYUNDAI_IONIQ_5, CAR.HYUNDAI_KONA_NON_SCC))
def test_kona_2022_friction_does_not_change_sibling_policies(monkeypatch, vehicle):
  from openpilot.starpilot.lateral import hkg_vehicle_shaping as shaping

  def unexpected(*_):
    raise AssertionError('Kona 2022 friction used by sibling policy')

  monkeypatch.setattr(shaping, 'get_kona_ev_2022_friction_threshold', unexpected)
  cp = interfaces[vehicle].get_non_essential_params(vehicle)
  controller = LatControlTorque(cp.as_reader(), interfaces[vehicle](cp), .01, controller_mode=ControllerMode.STARPILOT)
  threshold, scale = controller.starpilot_extension.policy.friction_context(.1, 0., 0., 120. / 3.6)
  assert threshold > 0. and scale > 0.


@pytest.mark.parametrize('alpha', (False, True))
@pytest.mark.parametrize('direction', (-1., 1.))
def test_ev6_highway_center_damping_preserves_controller_and_curve_response(monkeypatch, alpha, direction):
  from types import SimpleNamespace
  import math
  from openpilot.cereal import log
  from opendbc.car.vehicle_model import VehicleModel
  from openpilot.starpilot.lateral import hkg_vehicle_shaping as shaping

  cp = interfaces[CAR.KIA_EV6].get_non_essential_params(CAR.KIA_EV6)
  cp.openpilotLongitudinalControl, cp.pcmCruise = alpha, not alpha
  cs = SimpleNamespace(vEgo=26., steeringAngleDeg=0., steeringPressed=False)
  params = log.VehicleParameters.new_message(angleOffsetDeg=0., roll=0.)
  selected = shaping.KIA_EV6_HIGHWAY_CENTER_DAMPING
  assert selected == .08

  def run(taper, turn, mode=ControllerMode.STARPILOT):
    monkeypatch.setattr(shaping, 'KIA_EV6_HIGHWAY_CENTER_DAMPING', taper)
    controller = LatControlTorque(cp.as_reader(), interfaces[cp.carFingerprint](cp), .01, controller_mode=mode)
    vm = VehicleModel(cp)
    outputs, stages = [], []
    for tick in range(300):
      acceleration = direction * (1.5 if turn else .12 * math.sin(2 * math.pi * .4 * tick * .01))
      value, _, state = controller.update(True, cs, vm, params, False, acceleration / cs.vEgo**2, False, .386)
      assert math.isfinite(value) and abs(value) <= controller.steer_max + 1e-6
      outputs.append(value)
      stages.append((state.p, state.i, state.d, state.f))
    off, _, state = controller.update(False, cs, vm, params, False, 0., False, .386)
    assert off == 0. and not state.active
    return outputs, stages, controller.torque_params.to_dict(), (controller.pid.pos_limit, controller.pid.neg_limit, controller.steer_max)

  actual, baseline = run(selected, False), run(0., False)
  assert actual[1:] == baseline[1:]
  assert sum(x*x for x in actual[0]) < .88 * sum(x*x for x in baseline[0])
  assert all(a*b >= -1e-12 for a, b in zip(actual[0], baseline[0], strict=True))
  actual_turn, baseline_turn = run(selected, True), run(0., True)
  assert actual_turn[1:] == baseline_turn[1:]
  assert actual_turn[0] == pytest.approx(baseline_turn[0], rel=1e-6, abs=1e-8)
  assert run(selected, False, ControllerMode.STANDARD) == run(0., False, ControllerMode.STANDARD)


def test_ev6_taper_keeps_low_speed_and_substantial_curve_effect_small():
  from openpilot.starpilot.lateral import hkg_vehicle_shaping as shaping
  for speed in (0., 3., 26., 36.):
    for acceleration in (-2., -1., 0., 1., 2.):
      actual = shaping.get_kia_ev6_center_output_scale(acceleration, speed)
      speed_weight = shaping._kia_ev6_sigmoid((speed - shaping.KIA_EV6_CENTER_OUTPUT_TAPER_SPEED) /
                                              shaping.KIA_EV6_CENTER_OUTPUT_TAPER_SPEED_WIDTH)
      center_weight = shaping._kia_ev6_sigmoid((shaping.KIA_EV6_CENTER_OUTPUT_TAPER_LAT - abs(acceleration)) /
                                               shaping.KIA_EV6_CENTER_OUTPUT_TAPER_LAT_WIDTH)
      old = 1. - .14 * speed_weight * center_weight
      assert .78 <= actual <= old <= 1.
      if speed <= 3.:
        assert old - actual < .002
      if abs(acceleration) >= 1.:
        assert old - actual < .00002


def test_ioniq_5_n_profile_damping_unwind_and_controller_limits():
  from types import SimpleNamespace
  from openpilot.cereal import log
  from opendbc.car.vehicle_model import VehicleModel
  from opendbc.car.hyundai.values import CarControllerParams

  cp = interfaces[CAR.HYUNDAI_IONIQ_5_N].get_non_essential_params(CAR.HYUNDAI_IONIQ_5_N)
  assert cp.steerRatio == pytest.approx(12.4)
  assert cp.lateralTuning.torque.latAccelFactor == pytest.approx(3.25)
  assert cp.lateralTuning.torque.friction == pytest.approx(0.078)

  low_limits = CarControllerParams(cp, vEgoRaw=10.0)
  high_limits = CarControllerParams(cp, vEgoRaw=20.0)
  assert (low_limits.STEER_MAX, low_limits.STEER_DRIVER_ALLOWANCE, low_limits.STEER_THRESHOLD,
          low_limits.STEER_DELTA_UP, low_limits.STEER_DELTA_DOWN) == (384, 200, 220, 4, 5)
  assert (high_limits.STEER_MAX, high_limits.STEER_DELTA_UP, high_limits.STEER_DELTA_DOWN) == (384, 2, 3)

  controller = LatControlTorque(cp.as_reader(), interfaces[CAR.HYUNDAI_IONIQ_5_N](cp), 0.01, controller_mode=ControllerMode.STARPILOT)
  policy = controller.starpilot_extension.policy
  assert policy.profile == 'ioniq_5_n'
  assert controller.pid._k_i == ([0], [0.12])
  assert controller.pid._k_d[1] == [0.12, 0.10, 0.07]

  vm = VehicleModel(cp)
  params = log.VehicleParameters.new_message(angleOffsetDeg=0.0, roll=0.0)
  cs = SimpleNamespace(vEgo=10.0, steeringAngleDeg=0.0, steeringRateDeg=0.0, steeringPressed=False)
  controller.update(True, cs, vm, params, False, 0.0, False, 0.2)
  cs.steeringAngleDeg = -4.0
  _, _, moving = controller.update(True, cs, vm, params, False, 0.0, False, 0.2)
  assert moving.d < 0.0

  assert policy.feedforward_context(0.8, 1.5, -0.8, 1.9, 12.0) == pytest.approx(0.8)
  assert policy.feedforward_context(0.8, 0.4, -0.8, 0.9, 12.0) < 0.8
  assert policy.feedforward_context(0.8, 0.5, 0.7, 0.3, 12.0) < 0.8
  assert policy.feedforward_context(0.8, 1.5, 0.7, 1.3, 12.0) == pytest.approx(0.8)
  assert policy.jerk_deadzone(0.4, 0.6, 0.2, 10.0) > 0.10
  assert policy.jerk_deadzone(0.7, -0.3, 0.8, 25.0) >= 0.50
  policy.last_vm_sR = 14.94
  assert policy.steer_ratio_scale(22.0) == pytest.approx(12.6 / 14.94)
  policy.prev_output = 0.0
  first_out = policy.output_context(0.6, 0.5, 0.7, 0.3, 10.0)
  policy.prev_output = 0.0
  second_out = policy.output_context(0.6, 0.5, 0.1, 0.3, 10.0)
  assert first_out < second_out
  policy.hev_center_weight = 0.0
  policy.last_speed = 15.0
  policy.last_measurement = 0.0
  policy.last_setpoint = 0.0
  policy.steering_angle_deg = 0.0
  assert policy.pid_error_context(1.0) == pytest.approx(0.72)
  policy.last_speed = 3.5
  policy.last_measurement = -2.4
  policy.last_setpoint = -0.2
  policy.steering_angle_deg = 240.0
  assert abs(policy.pid_error_context(2.2)) < 0.72 * 2.2 * 0.35
  policy.steering_rate_deg = 100.0
  assert policy.output_context(1.0, -0.1, 2.0, -2.2, 3.5) == pytest.approx(0.50)
  policy.steering_rate_deg = 320.0
  assert policy.output_context(1.0, -0.1, 2.0, -2.2, 3.5) == pytest.approx(0.05)


def test_ioniq_5_n_sharp_turn_unwind_releases_torque_while_preserving_highway():
  from types import SimpleNamespace
  from openpilot.cereal import log
  from opendbc.car.vehicle_model import VehicleModel

  cp = interfaces[CAR.HYUNDAI_IONIQ_5_N].get_non_essential_params(CAR.HYUNDAI_IONIQ_5_N)
  vm = VehicleModel(cp)
  params = log.VehicleParameters.new_message(angleOffsetDeg=0.0, roll=0.0)
  controller = LatControlTorque(cp.as_reader(), interfaces[CAR.HYUNDAI_IONIQ_5_N](cp), 0.01, controller_mode=ControllerMode.STARPILOT)
  policy = controller.starpilot_extension.policy

  # 1. Sharp right-turn exit (~20 mph / 9.0 m/s, sa = -115 deg -> -85 deg):
  # Feedforward holds cleanly at apex, then sheds into-turn hold during corner exit while P assists unwind and D does not oppose wheel return.
  cs = SimpleNamespace(vEgo=9.0, steeringAngleDeg=-115.0, steeringRateDeg=0.0, steeringPressed=False)
  for _ in range(30):
    controller.update(True, cs, vm, params, False, 0.045, False, 0.2)

  hold_out, _, hold_state = controller.update(True, cs, vm, params, False, 0.045, False, 0.2)
  assert hold_out < -0.55

  # Corner exit: desired curvature drops over 0.30s while wheel begins unwinding from -115 -> -85 deg
  for step in range(30):
    frac = (step + 1) / 30.0
    curv = 0.045 - 0.033 * frac
    angle = -115.0 + 30.0 * frac
    cs = SimpleNamespace(vEgo=9.2, steeringAngleDeg=angle, steeringRateDeg=80.0, steeringPressed=False)
    unwind_out, _, unwind_state = controller.update(True, cs, vm, params, False, curv, False, 0.2)

  assert policy.sharp_turn_weight > 0.80
  assert unwind_state.f < 0.45 * hold_state.f
  assert unwind_state.p < -0.60
  assert unwind_state.d < 0.05
  assert unwind_out > -0.10

  # 2. Highway straight & highway curve invariance (vEgo >= 16.5 m/s or |sa| <= 28 deg):
  # Sharp-turn gate is identically zero and _k_d stays at exact baseline.
  hw_controller = LatControlTorque(cp.as_reader(), interfaces[CAR.HYUNDAI_IONIQ_5_N](cp), 0.01, controller_mode=ControllerMode.STARPILOT)
  hw_policy = hw_controller.starpilot_extension.policy
  for speed, sa, curv in [(29.0, -10.5, 0.0019), (22.0, 14.0, -0.0032), (17.5, -17.5, 0.0038), (12.0, -18.0, 0.0080)]:
    hw_cs = SimpleNamespace(vEgo=speed, steeringAngleDeg=sa, steeringRateDeg=8.0, steeringPressed=False)
    hw_controller.update(True, hw_cs, vm, params, False, curv, False, 0.2)
    assert hw_policy.sharp_turn_weight == 0.0
    assert hw_controller.pid._k_d[1] == [0.12, 0.10, 0.07]


def test_ioniq_5_n_flm_snapshot_and_native_refinement_compatibility():
  from types import SimpleNamespace
  from openpilot.cereal import log
  from opendbc.car import structs
  from opendbc.car.vehicle_model import VehicleModel
  from openpilot.starpilot.flm.controller_replay import (
    ControllerReplay,
    ReplayFrame,
    SelectedControllerFactory,
    capture_snapshot,
    final_command,
    frame_input,
    restore_snapshot,
  )
  from openpilot.starpilot.flm.native_baseline import checkpoint_native_valid, effective_native_valid
  from openpilot.starpilot.flm.runtime_adapter_registry import baseline_revision, baseline_values, selected_descriptor, selected_identity
  from openpilot.starpilot.flm.torque_adapter import Objective

  cp = interfaces[CAR.HYUNDAI_IONIQ_5_N].get_non_essential_params(CAR.HYUNDAI_IONIQ_5_N)
  factory = SelectedControllerFactory(cp.as_reader(), 'starpilot')
  live, live_vm = factory.create()
  probe, probe_vm = factory.create()

  # 1. Native torqued refinements below 3.0 (e.g. 2.85, within 0.7 * 3.25 = 2.275) must not be clobbered by update()
  live.update_torque_parameters(2.85, -0.04, 0.09)
  assert effective_native_valid(live, cp)
  before_tune = baseline_values(live)
  cs = SimpleNamespace(
    vEgo=9.5, aEgo=0.1, steeringAngleDeg=-110.0, steeringRateDeg=65.0, steeringTorque=-180.0,
    steeringPressed=False, standstill=False, steerFaultTemporary=False, steerFaultPermanent=False,
    leftBlinker=False, rightBlinker=False, gasPressed=False, brakePressed=False,
    gearShifter=structs.CarState.GearShifter.drive,
  )
  vp = log.VehicleParameters.new_message(angleOffsetDeg=0.0, roll=0.01)
  live.update(True, cs, live_vm, vp, False, 0.025, False, 0.2)
  assert baseline_values(live) == before_tune

  # 2. FLM ControllerReplay capture_snapshot + restore_snapshot + exact frame verification across sharp-turn unwind
  ctrl, ctrl_vm = factory.create()
  ci = interfaces[CAR.HYUNDAI_IONIQ_5_N](cp.as_reader())
  desc = selected_descriptor(cp.as_reader(), ci, ctrl)
  assert desc is not None
  base_rev = baseline_revision(desc, baseline_values(ctrl))
  ident = selected_identity(desc, '0' * 64, base_rev)
  replay = ControllerReplay(
    factory, Objective('feedforward', 'high', 'steady'),
    identity=ident, baseline_revision=base_rev, model_revision='test',
  )
  snap0 = capture_snapshot(ctrl, ctrl_vm)
  replay.checkpoint(100_000_000, snap0)
  for idx in range(6):
    t_ns = 100_000_000 + idx * 10_000_000
    step_cs = structs.CarState.new_message(
      vEgo=9.2, aEgo=0.0, steeringAngleDeg=-115.0 + idx * 8.0, steeringRateDeg=80.0, steeringTorque=-150.0,
      steeringPressed=False, standstill=False, steerFaultTemporary=False, steerFaultPermanent=False,
      leftBlinker=False, rightBlinker=False, gasPressed=False, brakePressed=False,
      gearShifter=structs.CarState.GearShifter.drive,
    )
    inputs = frame_input(
      step_cs, ctrl_vm, vp, active=True, safety_limited=False,
      desired_curvature=0.040 - idx * 0.005, curvature_limited=False, lat_delay=0.2,
    ) + (('control_frame', float(idx + 1)),)
    out, _, _ = ctrl.update(True, step_cs, ctrl_vm, vp, False, 0.040 - idx * 0.005, False, 0.2)
    assert checkpoint_native_valid(ctrl, cp, now_ns=t_ns)
    t_vals = (
      ('effective_factor', float(ctrl.torque_params.latAccelFactor)),
      ('offset', float(ctrl.torque_params.latAccelOffset)),
      ('friction', float(ctrl.torque_params.friction)),
      ('pid_pos_limit', float(ctrl.pid.pos_limit)),
      ('pid_neg_limit', float(ctrl.pid.neg_limit)),
    )
    replay.raw_mode = True
    tf = ReplayFrame(
      control_ns=t_ns, inputs=inputs, selected_command=final_command(out),
      applied_coordinates=(), frame_sequence=idx + 1,
      actual_lac_command=float(out), actual_stage_residual=(0.0, 0.0),
      torque_values=t_vals, controller_domain='torque',
      healthy=True, manual=False,
    )
    norm = replay.normalize(tf, _verify_only=True)
    assert norm is not None and norm.verified
  snap_after = capture_snapshot(ctrl, ctrl_vm)
  restore_snapshot(probe, probe_vm, snap_after)


@pytest.mark.parametrize('vehicle,profile,factor', CASES)
@pytest.mark.parametrize('alpha_long', [False, True])
def test_vehicle_tune_defaults_are_independent_of_longitudinal_choice(vehicle, profile, factor, alpha_long):
  cp = interfaces[vehicle].get_non_essential_params(vehicle)
  cp.openpilotLongitudinalControl = alpha_long
  assert default_selection(cp).mode == ControllerMode.STARPILOT
  raw = replace_mode(None, cp, ControllerMode.STANDARD)
  selected = selection_from_bytes(cp, raw)
  assert selected.mode == ControllerMode.STANDARD
  assert selected.source == 'saved'
  assert selected.policy == profile


@pytest.mark.parametrize('vehicle', [CAR.KIA_CARNIVAL_2025, CAR.KIA_CARNIVAL_HEV_4TH_GEN])
@pytest.mark.parametrize('alpha', [False, True])
def test_carnival_crawl_center_tune_reaches_selected_controller(monkeypatch, vehicle, alpha):
  from types import SimpleNamespace
  from openpilot.cereal import log
  from opendbc.car.vehicle_model import VehicleModel
  from opendbc.car import structs
  from openpilot.starpilot.lateral import hkg_vehicle_shaping as shaping

  cp = interfaces[vehicle].get_non_essential_params(vehicle)
  cp.openpilotLongitudinalControl, cp.pcmCruise = alpha, not alpha
  cs = SimpleNamespace(vEgo=2., steeringAngleDeg=8., steeringPressed=False, standstill=False,
                       steerFaultTemporary=False, steerFaultPermanent=False,
                       gearShifter=structs.CarState.GearShifter.drive)
  params = log.VehicleParameters.new_message(angleOffsetDeg=0., roll=0.)

  def run(enabled):
    controller = LatControlTorque(cp.as_reader(), interfaces[vehicle](cp), .01,
                                  controller_mode=ControllerMode.STARPILOT)
    with monkeypatch.context() as patch:
      if not enabled:
        patch.setattr(shaping, 'get_kia_carnival_crawl_center_weight', lambda *args: 0., raising=False)
      output, _, state = controller.update(True, cs, VehicleModel(cp), params, False, .001, False, .3)
    return controller, output, state

  baseline, old_output, old_state = run(False)
  candidate, new_output, new_state = run(True)
  policy = candidate.starpilot_extension.policy
  assert policy.hev_center_weight == pytest.approx(1.)
  assert abs(new_state.p) < abs(old_state.p)
  assert abs(new_output) < abs(old_output)
  assert candidate.pid.pos_limit == baseline.pid.pos_limit
  assert candidate.pid.neg_limit == baseline.pid.neg_limit
  assert candidate.torque_params.to_dict() == baseline.torque_params.to_dict()
  cs.steeringPressed = True
  controller, _, _ = run(True)
  assert controller.starpilot_extension.policy.hev_center_weight == 0.
  cs.steeringPressed = False
  cs.vEgo = 6.
  candidate, output, state = run(True)
  baseline, old_output, old_state = run(False)
  assert output == old_output and state.to_dict() == old_state.to_dict()


@pytest.mark.parametrize('speed,setpoint,future,expected', [
  (0., 0., 0., 0.), (.3, 0., 0., 0.), (1., 0., 0., 1.), (2., .1, -.1, 1.),
  (3., -.1, .1, 1.), (4., 0., 0., .5), (5., 0., 0., 0.), (30., 0., 0., 0.),
  (2., .3, 0., 0.), (2., 0., -.3, 0.), (2., -.2, .1, .5),
])
def test_carnival_crawl_center_tune_boundaries(speed, setpoint, future, expected):
  from openpilot.starpilot.lateral import hkg_vehicle_shaping as shaping
  assert shaping.get_kia_carnival_crawl_center_weight(speed, setpoint, future) == pytest.approx(expected)
  assert shaping.get_kia_carnival_crawl_center_weight(speed, -setpoint, -future) == pytest.approx(expected)
