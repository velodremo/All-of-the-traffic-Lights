from agents import Agent
import sys
"""
A non-learning traffic light agent based on the LQF algorithm. 
TODO add credits 
"""

class LQFAgent(Agent):
    def __init__(self, junction):
        super().__init__(junction)
        # self.lanes = junction.incoming_lanes + junction.outgoing_lanes
        self.north_south_lanes, self.east_west_lanes = self._parse_lanes(
            junction.incoming_lanes, junction.outgoing_lanes)
        self.cur_phase = None
        self.phase_length = 0

    def _parse_lanes(self, in_lanes, out_lanes):
        """
        parse lanes into north-south / east-west
        :return:
        """
        north_south = []
        east_west = []
        if self.junction.junction_id == '0':
            # do the incoming first
            for in_lane in in_lanes:
                marker = in_lane.split('_')[0]
                if marker == '1' or marker == '0l' or marker == '0r':
                    east_west.append(in_lane)
                elif marker == '0u' or marker == '0d':
                    north_south.append(in_lane)
                else:
                    print("unrecognized lane :c")
                    exit(2)

        elif self.junction.junction_id == '1':
            # incoming
            for in_lane in in_lanes:
                marker = in_lane.split('_')[0]
                if marker == '0' or marker == '1l' or marker == '1r':
                    east_west.append(in_lane)
                elif marker == '1u' or marker == '1d':
                    north_south.append(in_lane)
                else:
                    print("unrecognized lane :c")
                    exit(2)

        else:
            print("unrecognized agent :c")
            exit()

        return north_south, east_west

    def pick_phase(self, state):
        """
        pick a phase based on the LQF principle
        :return: TODO
        """
        if self.cur_phase is not None:
            if self.phase_length < 3:
                self.phase_length += 1
                return self.cur_phase
            if self.cur_phase in [1, 3]:
                self.cur_phase = (self.cur_phase + 1) % 4
                self.phase_length = 0
                return self.cur_phase

        sum_north_south = 0
        sum_east_west = 0
        for lane, num_cars in state.items():
            if lane in self.north_south_lanes:
                sum_north_south += num_cars
            elif lane in self.east_west_lanes:
                sum_east_west += num_cars
            else:
                print("warning: got value for unregistered lane for LQF: %s" % str(lane), file=sys.stderr)

        if sum_north_south > sum_east_west:
            if self.cur_phase == 0:
                self.phase_length += 1
                return self.cur_phase # north-south
            else:
                self.cur_phase = 3 # prepare transition
                self.phase_length = 0
                return self.cur_phase
        else:
            if self.cur_phase == 2:
                self.phase_length += 1
                return self.cur_phase # east-west
            else:
                self.cur_phase = 1 # prepare transition
                self.phase_length = 0
                return self.cur_phase

    def getAction(self, state):
        state = list(state.values())[0]
        return self.pick_phase(state)

    def update(self, prev_observation, action, new_observation, global_reward):
        pass