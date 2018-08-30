import numpy as np
from HParams import global_hparams
import math

class StateExtractorAbstract():
    def __init__(self, junction, back_end, logger=None):
        self.traffic_light_id = junction.traffic_light_id
        self.outgoing_lanes = junction.outgoing_lanes
        self.incoming_lanes = junction.incoming_lanes
        self.back_end = back_end
        self.logger = logger

    def extract_state(self, vehicle_list, action):
        raise NotImplementedError()


class StateExtractor(StateExtractorAbstract):
    NUM_FEATURES_PER_LANE = 5
    MAX_PHASE_DURATION = 100
    MAX_WAITING_TIME = 50

    def __init__(self, junction, num_actions, num_phases, back_end, logger=None):
        super().__init__(junction, back_end, logger)
        self.num_actions = num_actions
        self.num_phases = num_phases

    def extract_state(self, vehicle_list, action):
        """
        creates a vector representation of the state
        :param vehicle_list: a list of vehicles around the the junction/agent
        :param action: an action to preform
        :return: a feature vector
        """

        feat_per_lane = StateExtractor.NUM_FEATURES_PER_LANE

        # features, avg speed per lane, number of vehicles per lane
        stats_lanes = dict()

        for vehicle_id in vehicle_list:
            lane_num = self.back_end.vehicle_get_lane_id(vehicle_id)
            sp = self.back_end.vehicle_get_speed(vehicle_id)
            waiting_time = self.back_end.vehicle_get_waiting_time(vehicle_id)


            pos_on_lane = self.back_end.vehicle_get_lane_position(vehicle_id)
            if lane_num not in stats_lanes:
                # init new lane statistics
                lane_len = self.back_end.get_length("lane", lane_num)

                lane_dict = {"num_cars": 1, "cumo_speed": sp, "cumo_time": waiting_time,
                                "dir": lane_num in self.incoming_lanes,
                             "lane_length": lane_len, "car_pos": [pos_on_lane]}

                stats_lanes[lane_num] = lane_dict

                # set to be L - p in case of incoming edge
                if stats_lanes[lane_num]["dir"]:
                    stats_lanes[lane_num]["car_pos"] = [lane_len - pos_on_lane]

            else:
                # update statistics for previously seen lane
                stats_lanes[lane_num]["num_cars"] += 1
                stats_lanes[lane_num]["cumo_speed"] += sp
                stats_lanes[lane_num]["cumo_time"] += waiting_time

                # set to be L - p in case of incoming edge
                if stats_lanes[lane_num]["dir"]:
                    pos_on_lane = stats_lanes[lane_num]["lane_length"] - pos_on_lane

                stats_lanes[lane_num]["car_pos"] += [pos_on_lane]

        for lane in self.incoming_lanes + self.outgoing_lanes:
                if lane not in stats_lanes:
                    stats_lanes[lane] = {"num_cars": 0, "cumo_speed": 0,
                                         "cumo_time": 0,
                                "dir": 0 in self.incoming_lanes,
                             "lane_length": 0, "car_pos": [0]}

        # computing length of features per lanes
        lane_feature_len = (len(self.incoming_lanes) + len(
            self.outgoing_lanes)) * feat_per_lane

        feat_len = lane_feature_len + self.num_phases + 1

        feature_vec = np.zeros(feat_len)

        curr_phase = self.back_end.tl_get_phase(self.traffic_light_id)

        # setting current phase to 1 other phases to 0
        feature_vec[curr_phase] = 1
        # phase duration feature
        if action == curr_phase:
            cur_duration = self.back_end.tl_get_phase_passed_duration(self.traffic_light_id)
        else:
            cur_duration = 0
        feature_vec[self.num_phases] = cur_duration / StateExtractor.MAX_PHASE_DURATION

        i = self.num_phases + 1

        num_vehicles_on_road = len(vehicle_list) + 1
        for lane_id in self.incoming_lanes + self.outgoing_lanes:
            lane = stats_lanes[lane_id]
            # factor of outgoing lanes
            factor = 1

            if lane["dir"]:
                # factor of incoming lanes
                factor = 1
            if lane["num_cars"] == 0:
                i += 4
                continue
            # avg speed
            feature_vec[i] = factor * lane["cumo_speed"] / lane["num_cars"]
            feature_vec[i] = min(feature_vec[i] / self.back_end.lane_get_max_speed(lane_id), 1)
            i += 1
            # avg waiting time
            feature_vec[i] = factor * lane["cumo_time"] / lane["num_cars"]
            feature_vec[i] = min(feature_vec[i] / StateExtractor.MAX_WAITING_TIME, 1)
            i += 1
            # closest car
            feature_vec[i] = min(lane["car_pos"]) / global_hparams.OBSERVED_RADIUS

            i += 1
            # number of cars
            feature_vec[i] = factor * lane["num_cars"]
            feature_vec[i] = min(feature_vec[i] / num_vehicles_on_road, 1)


            i += 1


        multicalss_vec = np.zeros(feat_len * self.num_actions)

        multicalss_vec[action * feat_len: (action + 1) * feat_len] = feature_vec

        if self.logger is not None:
            self.logger["tl_"+self.traffic_light_id+"_phase"] = curr_phase

        return multicalss_vec


class StateExtractorCarCounter(StateExtractorAbstract):
    NUM_FEATURES_PER_LANE = 5
    MAX_PHASE_DURATION = 100
    MAX_WAITING_TIME = 50

    def __init__(self, junction, num_actions, num_phases, back_end, logger=None):
        super().__init__(junction, back_end, logger)
        self.num_actions = num_actions
        self.num_phases = num_phases

    def extract_state(self, vehicle_list, action):
        """
        creates a vector representation of the state
        :param vehicle_list: a list of vehicles around the the junction/agent
        :param action: an action to preform
        :return: a feature vector
        """

        lanes_num_cars = {lane_num: 0 for lane_num in self.incoming_lanes}

        for vehicle_id in vehicle_list:
            lane_num = self.back_end.vehicle_get_lane_id(vehicle_id)

            if lane_num in lanes_num_cars:
                lanes_num_cars[lane_num] += 1

        curr_phase = self.back_end.tl_get_phase(self.traffic_light_id)
        if self.logger is not None:
            self.logger["tl_" + self.traffic_light_id + "_phase"] = curr_phase

        return lanes_num_cars


class StateExtractorCompact(StateExtractorAbstract):
    NUM_FEATURES_PER_LANE = 4
    MAX_PHASE_DURATION = 100
    # MAX_NUM_CARS = 1000
    MAX_WAITING_TIME = 50

    def __init__(self, junction, num_actions, num_phases, back_end, logger=None):
        super().__init__(junction, back_end, logger)
        self.num_actions = num_actions
        self.num_phases = num_phases
        self.curr_duration = 0
        self.incoming_ns = list(filter(lambda lane_id: "d" in lane_id or "u" in lane_id, self.incoming_lanes))
        self.incoming_we = list(filter(lambda lane_id: lane_id not in self.incoming_ns,
                                       self.incoming_lanes)) # either r, l or 0_1

    def extract_state(self, vehicle_list, action):
        """
        creates a vector representation of the state
        :param vehicle_list: a list of vehicles around the the junction/agent
        :param action: an action to preform
        :return: a feature vector
        """

        feat_per_lane = StateExtractorCompact.NUM_FEATURES_PER_LANE

        # features, avg speed per lane, number of vehicles per lane
        stats_lanes = dict()

        for vehicle_id in vehicle_list:
            lane_num = self.back_end.vehicle_get_lane_id(vehicle_id)
            sp = self.back_end.vehicle_get_speed(vehicle_id)
            waiting_time = self.back_end.vehicle_get_waiting_time(vehicle_id)


            pos_on_lane = self.back_end.vehicle_get_lane_position(vehicle_id)
            if lane_num not in stats_lanes:
                # init new lane statistics
                lane_len = self.back_end.get_length("lane", lane_num)

                lane_dict = {"num_cars": 1, "cumo_speed": sp, "cumo_time": waiting_time,
                                "dir": lane_num in self.incoming_lanes,
                             "lane_length": lane_len, "car_pos": [pos_on_lane]}

                stats_lanes[lane_num] = lane_dict

                # set to be L - p in case of incoming edge
                if stats_lanes[lane_num]["dir"]:
                    stats_lanes[lane_num]["car_pos"] = [lane_len - pos_on_lane]

            else:
                # update statistics for previously seen lane
                stats_lanes[lane_num]["num_cars"] += 1
                stats_lanes[lane_num]["cumo_speed"] += sp
                stats_lanes[lane_num]["cumo_time"] += waiting_time

                # set to be L - p in case of incoming edge
                if stats_lanes[lane_num]["dir"]:
                    pos_on_lane = stats_lanes[lane_num]["lane_length"] - pos_on_lane

                stats_lanes[lane_num]["car_pos"] += [pos_on_lane]
        for lane in self.incoming_lanes + self.outgoing_lanes:
                if lane not in stats_lanes:
                    stats_lanes[lane] = {"num_cars": 0, "cumo_speed": 0,
                                         "cumo_time": 0,
                                "dir": 0 in self.incoming_lanes,
                             "lane_length": 0, "car_pos": [0]}

        # computing length of features per lanes
        lane_feature_len = 2 * feat_per_lane # green lanes and red lanes..

        feat_len = lane_feature_len + 3 # plus three phase features

        feature_vec = np.zeros(feat_len)

        curr_phase = self.back_end.tl_get_phase(self.traffic_light_id)

        # phase duration feature
        if curr_phase == action:
            feature_vec[0] = 1
            cur_duration = self.back_end.tl_get_phase_passed_duration(self.traffic_light_id)
        else:
            feature_vec[0] = 0
            cur_duration = 0

        feature_vec[1] = cur_duration / StateExtractor.MAX_PHASE_DURATION


        # transition phase feature
        if action in [1,3]:
            feature_vec[2] = 1

        num_vehicles_on_road = len(vehicle_list) + 1
        ns_features = np.zeros(StateExtractorCompact.NUM_FEATURES_PER_LANE)
        we_features = np.zeros(StateExtractorCompact.NUM_FEATURES_PER_LANE)
        i = 0
        for lane_id in self.incoming_ns:
            i = i % (StateExtractorCompact.NUM_FEATURES_PER_LANE)
            lane = stats_lanes[lane_id]
            # factor of outgoing lanes

            factor = 1
            if lane["num_cars"] == 0:
                continue
            # avg speed
            speed = min(feature_vec[i] / self.back_end.lane_get_max_speed(lane_id), 1)
            ns_features[i] += min(speed / self.back_end.lane_get_max_speed(lane_id), 1)
            i += 1
            # avg waiting time
            wait = factor * lane["cumo_time"] / lane["num_cars"]
            ns_features[i] += min(wait / StateExtractor.MAX_WAITING_TIME, 1)
            i += 1
            # closest car
            ns_features[i] = min(min(lane["car_pos"]) / global_hparams.OBSERVED_RADIUS, we_features[i])

            i += 1
            # number of cars
            num_cars = factor * lane["num_cars"]
            ns_features[i] += min(num_cars / num_vehicles_on_road, 1)

            i += 1

        for lane_id in self.incoming_we:
            i = i % (StateExtractorCompact.NUM_FEATURES_PER_LANE)
            lane = stats_lanes[lane_id]
            # factor of outgoing lanes

            factor = 1
            if lane["num_cars"] == 0:
                continue
            # avg speed
            speed = min(feature_vec[i] / self.back_end.lane_get_max_speed(lane_id), 1)
            we_features[i] += min(speed / self.back_end.lane_get_max_speed(lane_id), 1)
            i += 1
            # avg waiting time
            wait = factor * lane["cumo_time"] / lane["num_cars"]
            we_features[i] += min(wait / StateExtractor.MAX_WAITING_TIME, 1)
            i += 1
            # closest car
            we_features[i] += min(lane["car_pos"]) / global_hparams.OBSERVED_RADIUS

            i += 1
            # number of cars
            num_cars = factor * lane["num_cars"]
            we_features[i] += min(num_cars / num_vehicles_on_road, 1)

            i += 1
        if action in [1,2]:
            feature_vec[3: 3+StateExtractorCompact.NUM_FEATURES_PER_LANE] = we_features
            feature_vec[3 +
                        StateExtractorCompact.NUM_FEATURES_PER_LANE: 3 + 2*StateExtractorCompact.NUM_FEATURES_PER_LANE] = \
                ns_features
        else:
            feature_vec[3: 3 + StateExtractorCompact.NUM_FEATURES_PER_LANE] = ns_features
            feature_vec[3 +
                        StateExtractorCompact.NUM_FEATURES_PER_LANE: 3 + 2 * StateExtractorCompact.NUM_FEATURES_PER_LANE] = \
                we_features

        if self.logger is not None:
            self.logger["tl_"+self.traffic_light_id+"_phase"] = curr_phase
        return feature_vec
