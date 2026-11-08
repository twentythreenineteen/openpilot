import numpy as np

from opendbc.car.hyundai.values import CAR, HyundaiFlags

from openpilot.starpilot.lateral import hkg_vehicle_shaping as shaping

from openpilot.starpilot.lateral.hkg_torque_policy import HKGShapedTorquePolicy, supported_torque_cp

PROFILES = {
  'ioniq_5': (frozenset((CAR.HYUNDAI_IONIQ_5,)), shaping.IONIQ_5_BASE_LAT_ACCEL_FACTOR_MULT),
  'ioniq_5_n': (frozenset((CAR.HYUNDAI_IONIQ_5_N,)), 1),
  'ioniq_ev_old': (
    frozenset(
      (
        CAR.HYUNDAI_IONIQ_EV_LTD,
        CAR.HYUNDAI_IONIQ_EV_2020,
      )
    ),
    shaping.IONIQ_EV_OLD_BASE_LAT_ACCEL_FACTOR_MULT,
  ),
  'sonata': (frozenset((CAR.HYUNDAI_SONATA,)), 1),
  'elantra_non_scc': (
    frozenset(
      (
        CAR.HYUNDAI_ELANTRA_2022_NON_SCC,
        CAR.HYUNDAI_ELANTRA_HEV_2022_NON_SCC,
      )
    ),
    1,
  ),
  'kia_xceed': (frozenset((CAR.KIA_XCEED_PHEV,)), 1),
  'kia_niro_ev': (frozenset((CAR.KIA_NIRO_EV,)), 1),
  'kia_niro_phev_2022': (frozenset((CAR.KIA_NIRO_PHEV_2022,)), 1),
  'kia_stinger_2022': (frozenset((CAR.KIA_STINGER_2022,)), 1),
  'kia_forte': (
    frozenset(
      (
        CAR.KIA_FORTE,
        CAR.KIA_FORTE_2019_NON_SCC,
        CAR.KIA_FORTE_2021_NON_SCC,
      )
    ),
    shaping.KIA_FORTE_BASE_LAT_ACCEL_FACTOR_MULT,
  ),
  'kia_ev6': (frozenset((CAR.KIA_EV6,)), 1),
  'kia_carnival': (
    frozenset(
      (
        CAR.KIA_CARNIVAL_2025,
        CAR.KIA_CARNIVAL_HEV_4TH_GEN,
      )
    ),
    1,
  ),
  'tucson_4th_gen': (frozenset((CAR.HYUNDAI_TUCSON_4TH_GEN,)), 1),
  'kona_non_scc': (frozenset((CAR.HYUNDAI_KONA_NON_SCC,)), 1),
  'kona_ev_2022': (frozenset((CAR.HYUNDAI_KONA_EV_2022,)), 1),
}


def policy_for(cp):
  return next((key for key, (cars, _) in PROFILES.items() if supported_torque_cp(cp, cars)), None)


class HKGVehicleTorquePolicy(HKGShapedTorquePolicy):
  def __init__(self, parent, cp):
    self.profile = policy_for(cp)
    if self.profile is None:
      raise ValueError("Unsupported HKG vehicle torque profile")
    self.FACTOR_MULT = PROFILES[self.profile][1]
    self.canfd = bool(cp.flags & HyundaiFlags.CANFD)
    self.carnival_hev = cp.carFingerprint == CAR.KIA_CARNIVAL_HEV_4TH_GEN
    self.requested_curvature = 0.0
    self.future_lateral_accel = 0.0
    self.steering_pressed = False
    self.hev_center_weight = 0.0
    super().__init__(parent, cp)
    if self.profile == 'ioniq_5_n':
      from openpilot.common.pid import PIDController
      from openpilot.starpilot.lateral.gv70_shaping import GENESIS_GV70_MEASUREMENT_DAMPING_SPEED_BP
      from openpilot.starpilot.lateral.torque_shaping import INTERP_SPEEDS, KP_INTERP
      self.base_kd_bp = list(GENESIS_GV70_MEASUREMENT_DAMPING_SPEED_BP)
      self.base_kd_v = [0.12, 0.10, 0.07]
      self.prev_output = 0.0
      self.steering_angle_deg = 0.0
      self.steering_rate_deg = 0.0
      self.last_speed = 0.0
      self.last_setpoint = 0.0
      self.last_measurement = 0.0
      self.last_roll = 0.0
      self.filtered_roll = None
      self.last_vm_sR = float(cp.steerRatio)
      parent.pid = PIDController([INTERP_SPEEDS, KP_INTERP], 0.12,
                                 [self.base_kd_bp, list(self.base_kd_v)],
                                 rate=1 / self.dt)
      parent.update_limits()

  supported_cp = staticmethod(lambda cp: policy_for(cp) is not None)

  def steer_ratio_scale(self, speed):
    if self.profile == 'ioniq_5_n':
      target = float(np.interp(max(speed, 0.0), [8.0, 22.0], [13.2, 12.6]))
      raw_sr = max(getattr(self, 'last_vm_sR', 13.0), 10.0)
      return float(np.clip(target / raw_sr, 0.80, 1.06))
    return 1.0

  def update(self, active, cs, vm, params, safety_limited, curvature, curvature_limited, delay):
    self.requested_curvature = curvature
    self.future_lateral_accel = curvature * cs.vEgo ** 2
    self.steering_pressed = bool(cs.steeringPressed)
    if self.profile == 'ioniq_5_n':
      self.steering_angle_deg = float(getattr(cs, 'steeringAngleDeg', 0.0))
      self.steering_rate_deg = abs(float(getattr(cs, 'steeringRateDeg', 0.0)))
      self.last_speed = float(cs.vEgo)
      self.last_vm_sR = float(getattr(vm, 'sR', 13.0))
      raw_roll = float(getattr(params, 'roll', 0.0))
      self.last_roll = raw_roll
      if self.filtered_roll is None:
        self.filtered_roll = raw_roll
      elif abs(float(cs.steeringAngleDeg)) < 18.0 and abs(self.future_lateral_accel) < 0.75:
        self.filtered_roll += 0.005 * (raw_roll - self.filtered_roll)
      if self.parent.torque_params.latAccelFactor < 3.0:
        self.parent.torque_params.latAccelFactor = 3.25
        self.parent.update_limits()
      if not active:
        self.prev_output = 0.0
      else:
        if not cs.steeringPressed and abs(self.future_lateral_accel) > 0.65 and self.parent.pid.i * self.future_lateral_accel > 0.0:
          self.parent.pid.i *= 0.992
    return super().update(active, cs, vm, params, safety_limited, curvature, curvature_limited, delay)

  def feedforward_context(self, value, setpoint, jerk, measurement, speed):
    if self.profile == 'ioniq_5':
      return value * shaping.get_ioniq_5_ff_scale(setpoint, jerk, speed) * shaping.get_ioniq_5_center_taper_scale(setpoint, speed)
    if self.profile == 'ioniq_ev_old':
      return value * shaping.get_ioniq_ev_old_ff_scale(setpoint, jerk, speed) * shaping.get_ioniq_ev_old_center_taper_scale(setpoint, speed)
    if self.profile == 'sonata':
      return value * shaping.get_sonata_ff_scale(setpoint, jerk, speed) * shaping.get_sonata_center_taper_scale(setpoint, speed)
    if self.profile == 'elantra_non_scc':
      return value * shaping.get_elantra_non_scc_ff_scale(setpoint, jerk, speed)
    if self.profile == 'kia_xceed':
      return value * shaping.get_kia_xceed_ff_scale(setpoint, jerk, speed) * shaping.get_kia_xceed_center_taper_scale(setpoint, speed)
    if self.profile == 'kia_forte':
      return value * shaping.get_kia_forte_ff_scale(setpoint, jerk, speed) * shaping.get_kia_forte_center_taper_scale(setpoint, speed)
    if self.profile == 'kia_ev6':
      return value * shaping.get_kia_ev6_ff_scale(setpoint, jerk, speed) * shaping.get_kia_ev6_center_taper_scale(setpoint, speed)
    if self.profile == 'kia_carnival':
      return value * shaping.get_kia_carnival_unwind_ff_scale(setpoint, measurement, jerk, speed)
    if self.profile == 'ioniq_5_n':
      self.last_setpoint = float(setpoint)
      self.last_measurement = float(measurement)
      fade = float(np.interp(speed, [0.5, 2.5], [0.0, 1.0]))
      raw_roll_comp = float(getattr(self, 'last_roll', 0.0) * 9.81) * fade
      safe_roll = getattr(self, 'filtered_roll', None)
      if safe_roll is None:
        safe_roll = getattr(self, 'last_roll', 0.0)
      mount_bias = 0.038 if safe_roll > 0.015 else 0.0
      safe_roll_comp = float(np.clip((safe_roll - mount_bias) * 9.81, -0.14, 0.14)) * fade
      value = value + raw_roll_comp - safe_roll_comp
      if setpoint * jerk > 0.0 and abs(setpoint) < 1.4:
        turn_in_weight = min(abs(jerk) / 0.8, 1.0) * (1.0 - abs(setpoint) / 1.4)
        return value * (1.0 - 0.18 * turn_in_weight)
      if abs(setpoint) < 0.65 and setpoint * jerk < -0.18 and abs(measurement) - abs(setpoint) > 0.18:
        return value * (1.0 - 0.45 * (1.0 - shaping.get_kia_carnival_unwind_ff_scale(setpoint, measurement, jerk, speed)))
      return value
    return value

  def friction_context(self, setpoint, jerk, measurement, speed):
    base = shaping.get_hkg_canfd_base_friction_threshold(speed) if self.canfd else shaping.get_standard_friction_threshold(speed)
    if self.profile == 'ioniq_5':
      return shaping.get_ioniq_5_friction_threshold(speed, setpoint, jerk), 1 + (
        shaping.get_ioniq_5_friction_scale(speed, setpoint, jerk) - 1
      ) * shaping.get_ioniq_5_center_taper_scale(setpoint, speed)
    if self.profile == 'kia_niro_ev' and not self.steering_pressed:
      return shaping.get_kia_niro_ev_friction_threshold(speed, setpoint, jerk, self.future_lateral_accel), 1
    if self.profile == 'kia_niro_phev_2022':
      return shaping.get_kia_niro_phev_2022_friction_threshold(speed, setpoint, jerk), 1
    if self.profile == 'kia_stinger_2022':
      return shaping.get_kia_stinger_2022_friction_threshold(
        speed, setpoint, jerk, self.future_lateral_accel, self.steering_pressed or self.previous_pressed), 1
    if self.profile == 'kia_forte':
      return shaping.get_kia_forte_friction_threshold(speed, setpoint, jerk), 1
    if self.profile == 'kia_ev6':
      return shaping.get_kia_ev6_friction_threshold(speed, setpoint, jerk), 1 + (
        shaping.get_kia_ev6_friction_scale(speed, setpoint, jerk) - 1
      ) * shaping.get_kia_ev6_center_taper_scale(setpoint, speed)
    if self.profile in ('kia_carnival', 'ioniq_5_n'):
      self.hev_center_weight = (shaping.get_kia_carnival_hev_center_weight(
        speed, setpoint, measurement, jerk, self.future_lateral_accel)
        if (self.carnival_hev or self.profile == 'ioniq_5_n') and not self.steering_pressed and not self.previous_pressed else 0.0)
      threshold = shaping.get_kia_carnival_friction_threshold(speed, setpoint, jerk)
      return (threshold + shaping.KIA_CARNIVAL_HEV_CENTER_FRICTION_INCREMENT * self.hev_center_weight,
              shaping.get_kia_carnival_friction_center_fade_scale(setpoint, speed))
    if self.profile == 'tucson_4th_gen':
      return shaping.get_tucson_4th_gen_friction_threshold(speed, setpoint, jerk), 1
    if self.profile == 'kona_non_scc':
      return shaping.get_kona_non_scc_friction_threshold(speed, setpoint, jerk), 1
    if self.profile == 'kona_ev_2022':
      return shaping.get_kona_ev_2022_friction_threshold(speed, setpoint), 1
    return (base, 1.0)

  def pid_error_context(self, error):
    scale = 0.72 if self.profile == 'ioniq_5_n' else 1.0
    if self.profile == 'ioniq_5_n':
      from openpilot.starpilot.lateral.torque_shaping import INTERP_SPEEDS, KP_INTERP, LOW_SPEED_X, LOW_SPEED_Y, MIN_SPEED
      meas = getattr(self, 'last_measurement', 0.0)
      sp = getattr(self, 'last_setpoint', 0.0)
      sa = getattr(self, 'steering_angle_deg', 0.0)
      spd = max(float(getattr(self, 'last_speed', 15.0)), MIN_SPEED)
      lsf = (float(np.interp(spd, LOW_SPEED_X, LOW_SPEED_Y)) / spd) ** 2
      kp = max(float(np.interp(spd, INTERP_SPEEDS, KP_INTERP)), 0.1)
      # In openpilot, sa > 0 is Left (meas < 0) and sa < 0 is Right (meas > 0)
      effective_meas = meas if abs(meas) >= 0.25 else (-sa * 0.015)
      is_unwinding = (abs(meas) > 0.30 or abs(sa) > 20.0) and error * effective_meas < 0.0 and abs( meas ) >= abs(sp)
      if is_unwinding:
        lsf_ratio = (kp + lsf) / 1.8
        gap = max(abs(meas) - abs(sp), abs(sa) * 0.012)
        unwind_weight = min(gap / 0.75, 1.0)
        attenuated = (1.0 + 0.18 * lsf_ratio) / (1.0 + lsf_ratio)
        scale *= 1.0 - unwind_weight * (1.0 - attenuated)
      elif spd < 4.5:
        # Prevent 394x low-speed KP+LSF saturation during crawl-speed (<10 mph) intersection turns
        crawl_scale = float(np.interp(spd, [1.0, 2.5, 4.5], [0.42, 0.62, 1.0]))
        scale *= crawl_scale
    return error * scale * (1.0 - shaping.KIA_CARNIVAL_HEV_CENTER_ERROR_REDUCTION * self.hev_center_weight)

  def jerk_deadzone(self, setpoint, jerk, measurement, speed):
    if self.profile == 'ioniq_5':
      return shaping.get_ioniq_5_friction_jerk_deadzone(speed, setpoint)
    if self.profile == 'kia_carnival':
      return shaping.get_kia_carnival_friction_jerk_deadzone(speed, setpoint, jerk)
    if self.profile == 'ioniq_5_n':
      base_dz = shaping.get_kia_carnival_friction_jerk_deadzone(speed, setpoint, jerk)
      if abs(setpoint) >= 0.35 and setpoint * measurement > 0.0:
        curve_dz = float(np.interp(abs(setpoint), [0.35, 0.70, 1.80, 2.60], [0.22, 0.52, 0.42, 0.18]))
        base_dz = max(base_dz, curve_dz)
      if setpoint * jerk > 0.0 and abs(setpoint) < 0.85:
        return max(base_dz, 0.25 * (1.0 - abs(setpoint) / 0.85))
      return base_dz
    return 0.0

  def output_context(self, value, setpoint, jerk, measurement, speed):
    if self.profile == 'ioniq_5':
      limit = shaping.get_ioniq_5_low_speed_output_limit(
        setpoint, jerk, speed, self.requested_curvature if not self.steering_pressed and not self.previous_pressed else 0.0, value)
      return float(np.clip(value, -limit, limit))
    if self.profile == 'kia_niro_phev_2022':
      return value * shaping.get_kia_niro_phev_2022_center_taper_scale(setpoint, speed)
    if self.profile == 'kia_stinger_2022':
      return value * shaping.get_kia_stinger_2022_center_taper_scale(setpoint, speed)
    if self.profile == 'kia_ev6':
      return (value * shaping.get_kia_ev6_low_speed_center_taper_scale(setpoint, speed) *
              shaping.get_kia_ev6_center_output_scale(setpoint, speed, self.future_lateral_accel))
    if self.profile == 'kia_carnival':
      return (
        value
        * shaping.get_kia_carnival_center_taper_scale(setpoint, speed)
        * shaping.get_kia_carnival_unwind_output_scale(
          setpoint, measurement, jerk, speed,
          output_torque=value if self.carnival_hev and not self.steering_pressed and not self.previous_pressed else None)
        * shaping.get_kia_carnival_highway_transition_output_scale(setpoint, jerk, speed)
      )
    if self.profile == 'ioniq_5_n':
      turn_in_scale = 1.0
      if setpoint * jerk > 0.0 and abs(setpoint) < 1.25 and abs(jerk) > 0.25:
        turn_in_scale = 1.0 - 0.10 * min((abs(jerk) - 0.25) / 0.65, 1.0) * (1.0 - abs(setpoint) / 1.25)
      scaled = (
        value
        * turn_in_scale
        * shaping.get_kia_carnival_center_taper_scale(setpoint, speed)
        * shaping.get_kia_carnival_highway_transition_output_scale(setpoint, jerk, speed)
      )
      sa = getattr(self, 'steering_angle_deg', 0.0)
      eff_meas = measurement if abs(measurement) >= 0.25 else (-sa * 0.015)
      if (abs(measurement) > 0.35 or abs(sa) > 22.0) and abs(measurement) >= abs(setpoint) and scaled * eff_meas < 0.0:
        sr = getattr(self, 'steering_rate_deg', 0.0)
        max_rev = float(np.interp(sr, [130.0, 230.0, 320.0], [0.50, 0.16, 0.05]))
        if eff_meas > 0.0:
          scaled = max(float(scaled), -max_rev)
        else:
          scaled = min(float(scaled), max_rev)
      elif not self.steering_pressed and not self.previous_pressed and abs(setpoint) >= 0.35 and setpoint * measurement > 0.0:
        rc = 0.05 if setpoint * jerk < -0.35 else 0.11
        alpha = self.dt / (rc + self.dt)
        scaled = self.prev_output + alpha * (scaled - self.prev_output)
      self.prev_output = float(scaled)
      return float(scaled)
    if self.profile == 'tucson_4th_gen':
      return value * shaping.get_tucson_4th_gen_center_taper_scale(setpoint, speed)
    if self.profile == 'kona_non_scc':
      value *= shaping.get_kona_non_scc_center_taper_scale(setpoint, speed)
      if value * setpoint > 0 or setpoint * jerk < 0:
        value *= shaping.get_kona_non_scc_highway_transition_output_scale(setpoint, jerk, speed)
      return value
    if self.profile == 'kona_ev_2022':
      return value * shaping.get_kona_ev_2022_center_output_scale(setpoint, speed)
    return value
