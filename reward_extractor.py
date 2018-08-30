from state_representation import StateExtractorAbstract
import numpy as np
from collections import defaultdict
from HParams import global_hparams as gh


class RewardExtractor():
    def restart(self):
        self.commulative_vehicle_time = {}
        self.phase = self.backend.tl_get_phase(self.traffic_light_id)

    def _calc_delay(self, vid):
        speed = self.backend.vehicle_get_speed(vid)
        allowed_speed = self.backend.vehicle_get_allowed_speed(vid)
        return 1 - (speed / allowed_speed)

    def _calc_wait(self, vid):
        """
        Return normalized wait score per vehicle
        :param vid:
        :return:
        """
        max_wait_time = 50
        # wait = self.backend.vehicle_get_waiting_time(vid)
        if vid in self.commulative_vehicle_time:
            wait = self.commulative_vehicle_time[vid]
        else:
            wait = 0
        res = min(1, wait/max_wait_time)
        return res

    def _calc_emergency_brake(self, vid):
        accel = self.backend.vehicle_get_acceleration(vid)
        return accel <= -4.5

    def __init__(self, junction, backend, logger, delay_w=0.3, flicker_w=0.1, teleport_w=0,
                 wait_w=0.3, emergency_brake_w=0.2):
        """
        Initialize a reward extractor for a specific junction.
        :param junction: A junction object
        :param delay_w: weighting factor for delays (deviation from max
        allowed speed) per car
        :param flicker_w: weighting factor for flickering (change of traffic
        light phase)
        :param teleport_w: weighting factor for teleporting (accidents)
        :param wait_w: weighting factor for car waiting time
        :param emergency_brake_w:

        Credit for the reward function - v.d. Pol, Oliehoek, Coordinated Deep
        Reinforcement
        Learners for Traffic Light Control
        https://www.fransoliehoek.net/docs/VanDerPol16LICMAS.pdf
        """

        self.traffic_light_id = junction.traffic_light_id
        self.outgoing_lanes = junction.outgoing_lanes
        self.incoming_lanes = junction.incoming_lanes
        self.junction = junction
        self.backend = backend

        self.phase = self.backend.tl_get_phase(self.traffic_light_id)
        self.delay_w = delay_w
        self.flicker_w = flicker_w
        self.teleport_w = teleport_w
        self.wait_w = wait_w
        self.delay_w = delay_w
        self.emergency_brake_w = emergency_brake_w
        self.logger = logger
        self.commulative_vehicle_time = {}

    def get_reward(self, vehicle_list):
        """
        Calculate and return a reward for a given simulation step.
        :param vehicle_list: A list of vehicle ids to
        :return: A weighted sum of the accumulated car waiting time, delayed
        time (deviation from max allowed speed), emergency brakes (
        decceleration over 4.5 m/s^2) and flickering (change of traffic
        light phase)
        """

        try:
            teleport_list = set(self.backend.get_starting_teleport_id_list())

        except NotImplementedError:
            teleport_list = []

        # get the current phase, check if its different than the last one (
        # flicker factor)
        cur_phase = self.backend.tl_get_phase(self.traffic_light_id)
        flicker = (cur_phase != self.phase)
        # if there are no cars return 0
        delay = wait = emergency_brake = teleports = 0

        if not vehicle_list:
            normalize_factor = 1
        else:
            # iterate over all the vehicles and calc the sums
            normalize_factor = 1e-8
            for vid in vehicle_list:
                # update the "time in junction" of the vehicle
                if vid not in self.commulative_vehicle_time:
                    self.commulative_vehicle_time[vid] = 1
                else:
                    self.commulative_vehicle_time[vid] += 1

                priority_factor = 1
                # calculate all metrics but accidents on incoming lanes
                if self.backend.vehicle_get_lane_id(vid) in self.incoming_lanes:
                    delay += self._calc_delay(vid) * priority_factor
                    wait += self._calc_wait(vid) * priority_factor
                    emergency_brake += self._calc_emergency_brake(vid) * priority_factor
                    normalize_factor += priority_factor
                teleports += int(vid in teleport_list)


        # normalize reward per vehicle batch size
        delay /= normalize_factor
        wait /= normalize_factor
        emergency_brake /= normalize_factor


        # log non-weighted individual rewards
        if self.logger is not None:
            self.logger["j_"+ self.junction.junction_id+'_delay'] = delay
            self.logger["j_"+ self.junction.junction_id+'_wait'] = wait
            self.logger["j_"+ self.junction.junction_id+'_emergency_brake'] = emergency_brake
            self.logger["j_"+ self.junction.junction_id+'_flicker'] = flicker
            self.logger["j_" + self.junction.junction_id + '_teleport'] = teleports
            self.logger["j_" + self.junction.junction_id + "_num_cars"] = normalize_factor



        # weigh the rewards and calculate and log total reward
        delay *= self.delay_w
        wait *= self.wait_w
        emergency_brake *= self.emergency_brake_w
        flicker *= self.flicker_w
        teleports *= self.teleport_w
        total_w_reward = -(delay + wait + emergency_brake + flicker + teleports)
        if self.logger is not None:
            self.logger["j_"+ self.junction.junction_id+'_total_w'] = total_w_reward

        # update the current phase
        self.phase = cur_phase

        return total_w_reward


class RewardExtractorWait:
    def __init__(self, junction, backend, logger):
        """
        Initialize a reward extractor for a specific junction.
        :param junction: A junction object
        :param delay_w: weighting factor for delays (deviation from max
        allowed speed) per car
        :param flicker_w: weighting factor for flickering (change of traffic
        light phase)
        :param teleport_w: weighting factor for teleporting (accidents)
        :param wait_w: weighting factor for car waiting time
        :param emergency_brake_w:

        """
        self.traffic_light_id = junction.traffic_light_id
        self.outgoing_lanes = junction.outgoing_lanes
        self.incoming_lanes = junction.incoming_lanes
        self.junction = junction
        self.backend = backend

        self.logger = logger
        self.prev_wait = 0

    def get_reward(self, vehicle_list):
        """
        Calculate and return a reward for a given simulation step.
        :param vehicle_list: A list of vehicle ids to
        :return:
        """

        wait = 0

        for vehicle_id in vehicle_list:
            if self.backend.vehicle_get_lane_id(vehicle_id) in self.incoming_lanes:
                wait += self.backend.vehicle_get_waiting_time(vehicle_id)

        delta_wait = wait - self.prev_wait

        self.prev_wait = wait

        total_w_reward = delta_wait

        if self.logger is not None:
            self.logger["j_" + self.junction.junction_id + '_delay'] = 0
            self.logger["j_" + self.junction.junction_id + '_wait'] = wait
            self.logger["j_" + self.junction.junction_id + '_emergency_brake'] = 0
            self.logger["j_" + self.junction.junction_id + '_flicker'] = 0
            self.logger["j_" + self.junction.junction_id + '_total_w'] = total_w_reward

        return total_w_reward
