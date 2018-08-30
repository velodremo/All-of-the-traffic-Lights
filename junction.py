# import libsumo


class Junction():
    """
    Value class that represents a junction and holds relevant information for querying libsumo.
    """
    def __init__(self, junction_id, traffic_light_id, sumo_backend):
        self.sumo_backend = sumo_backend
        self.junction_id = junction_id
        self.traffic_light_id = traffic_light_id
        self.outgoing_lanes, self.incoming_lanes = self._extract_lanes()

    def __hash__(self):
        return self.junction_id.__hash__()

    def _extract_lanes(self):
        lanes = self.sumo_backend.get_id_list("lane")
        lanes = filter(lambda l: ':' not in l, lanes)
        outgoing_lanes = []
        ingoing_lanes = []
        for laneID in lanes:
            if laneID.startswith(self.junction_id+"_"):
                outgoing_lanes.append(laneID)
            elif laneID.endswith(self.junction_id+"_0"):
                ingoing_lanes.append(laneID)
        return outgoing_lanes, ingoing_lanes
