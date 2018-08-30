import sys

try:
    import libsumo
except:
    print("failed to import libsumo")

try:
    sys.path.append("/opt/sumo/tools")
    import traci
except:
    print("failed to import traci")

# used for optimizing Traci performance by avoiding multiple similar calls
def cache_results(func):
    def memoized(*args, **kwargs):
        key = str(args) + str(kwargs) + str(func)
        if key not in SumoBackend._CACHE:
            res = func(*args, **kwargs)
            SumoBackend._CACHE[key] = res
        else:
            res = SumoBackend._CACHE[key]
        return res
    return memoized


_BACKEND = [None]


def set_backend(backend_type):
    if backend_type == "libsumo":
        _BACKEND[0] = LibsumoBackend()
    if backend_type in ["traci", "traci-gui"]:
        _BACKEND[0] = TraciBackend()


def get_backend():
    return _BACKEND[0]


class SumoBackend(object):
    _CACHE = {}

    def __init__(self):
        self.phase_durations = {}
        self.current_phases = {}

    def _manage_phase_durations(self, tl_id, phase):
        if tl_id not in self.current_phases:
            self.phase_durations[tl_id] = 0
            self.current_phases[tl_id] = phase
        elif phase == self.current_phases[tl_id]:
            self.phase_durations[tl_id] += 1
        else:
            self.current_phases[tl_id] = phase
            self.phase_durations[tl_id] = 0

    def close(self):
        raise NotImplementedError()

    def start(self, cmd):
        raise NotImplementedError()

    def simulation_step(self):
        raise NotImplementedError()

    def get_id_list(self, object_type):
        raise NotImplementedError()

    def tl_set_phase(self, tl_id, phase):
        raise NotImplementedError()

    def vehicle_get_lane_id(self, id):
        raise NotImplementedError()

    def tl_get_phase(self, tl_id):
        raise NotImplementedError()

    def tl_get_phase_duration(self, tl_id):
        raise NotImplementedError()

    def tl_get_controlled_lanes(self, tl_id):
        raise NotImplementedError()

    def vehicle_get_lane_position(self, id):
        raise NotImplementedError()

    def get_length(self, object_type, id):
        raise NotImplementedError()

    def vehicle_get_speed(self, id):
        raise NotImplementedError()

    def vehicle_get_allowed_speed(self, id):
        raise NotImplementedError()

    def vehicle_get_waiting_time(self, id):
        raise NotImplementedError()

    def vehicle_get_acceleration(self, id):
        raise NotImplementedError()

    def get_position(self, object_type, id):
        raise NotImplementedError()

    def get_starting_teleport_number(self):
        raise NotImplementedError()

    def lane_get_max_speed(self, id):
        raise NotImplementedError()

    def vehicle_get_type(self, id):
        raise NotImplementedError()

    def lane_get_shape(self, id):
        raise NotImplementedError()

    def lane_get_car_count(self, id):
        raise NotImplementedError()

    def lane_get_vehicle_count(self, id):
        raise NotImplementedError()

    def lane_get_occupancy(self, id):
        raise NotImplementedError()

    def get_starting_teleport_id_list(self):
        raise NotImplementedError()

    def tl_get_phase_passed_duration(self, tl_id):
        if tl_id not in self.phase_durations:
            return 0
        else:
            return self.phase_durations[tl_id]


class LibsumoBackend(SumoBackend):
    def vehicle_get_lane_id(self, id):
        return libsumo.vehicle_getLaneID(id)

    def tl_get_controlled_lanes(self, tl_id):
        return libsumo.trafficlight_getControlledLanes(tl_id)

    def vehicle_get_acceleration(self, id):
        return libsumo.vehicle_getAcceleration(id)

    def get_id_list(self, object_type):
        type_to_func = {"vehicle": libsumo.vehicle_getIDList,
                        "lane": libsumo.lane_getIDList}
        func = type_to_func[object_type]
        return func()

    def tl_get_phase_duration(self, tl_id):
        return libsumo.trafficlight_getPhaseDuration(tl_id)

    def tl_get_phase(self, tl_id):
        return libsumo.trafficlight_getPhase(tl_id)

    def get_starting_teleport_number(self):
        return 0

    def vehicle_get_allowed_speed(self, id):
        return libsumo.vehicle_getAllowedSpeed(id)

    def vehicle_get_waiting_time(self, id):
        return libsumo.vehicle_getWaitingTime(id)

    @cache_results  # necessary for the case of multiple set phases in one step for the same traffic light
    def tl_set_phase(self, tl_id, phase):
        self._manage_phase_durations(tl_id, phase)
        return libsumo.trafficlight_setPhase(tl_id, phase)

    def vehicle_get_lane_position(self, id):
        return libsumo.vehicle_getLanePosition(id)

    def close(self):
        libsumo.simulation_close()

    def start(self, cmd):
        libsumo.simulation.load(cmd)

    def get_length(self, object_type, id):
        type_to_func = {
            "lane": libsumo.lane_getLength
        }
        return type_to_func[object_type](id)

    def vehicle_get_speed(self, id):
        return libsumo.vehicle_getSpeed(id)

    def simulation_step(self):
        libsumo.simulation_step()
        SumoBackend._CACHE = {}

    def get_position(self, object_type, id):
        type_to_method = {"vehicle": libsumo.vehicle_getPosition,
                          "junction": libsumo.junction_getPosition}
        pos = type_to_method[object_type](id)
        return (pos.x, pos.y)

    def lane_get_max_speed(self, id):
        return libsumo.lane_getMaxSpeed(id)

    def vehicle_get_type(self, id):
        return libsumo.vehicle_getTypeID(id)

    def lane_get_shape(self, id):
        return libsumo.lane_getShape(id)

    def lane_get_vehicle_count(self, id):
        return libsumo.lane_getLastStepVehicleNumber(id)

    def lane_get_occupancy(self, id):
        return libsumo.lane_getLastStepOccupancy(id)




class TraciBackend(SumoBackend):
    @cache_results
    def get_starting_teleport_id_list(self):
        return traci.simulation.getStartingTeleportIDList()

    @cache_results
    def lane_get_vehicle_count(self, id):
        return traci.lane.getVehicleNumber(id)

    @cache_results
    def lane_get_shape(self, id):
        return traci.lane.getShape(id)

    @cache_results
    def lane_get_occupancy(self, id):
        return traci.lane.getLastStepOccupancy(id)

    def lane_get_car_count(self, id):
        pass

    @cache_results
    def vehicle_get_type(self, id):
        return traci.vehicle.getTypeID(id)

    @cache_results
    def vehicle_get_lane_id(self, id):
        return traci.vehicle.getLaneID(id)

    @cache_results
    def tl_get_controlled_lanes(self, tl_id):
        return traci.trafficlight.getControlledLanes(tl_id)

    @cache_results
    def vehicle_get_acceleration(self, id):
        return traci.vehicle.getAccel(id)

    @cache_results
    def get_id_list(self, object_type):
        type_to_func = {"vehicle": traci.vehicle.getIDList,
                        "lane": traci.lane.getIDList}
        func = type_to_func[object_type]
        return func()

    @cache_results
    def tl_get_phase_duration(self, tl_id):
        return traci.trafficlight.getPhaseDuration(tl_id)

    @cache_results
    def tl_get_phase(self, tl_id):
        return traci.trafficlight.getPhase(tl_id)

    @cache_results
    def get_starting_teleport_number(self):
        return traci.simulation.getStartingTeleportNumber()

    @cache_results
    def vehicle_get_allowed_speed(self, id):
        return traci.vehicle.getAllowedSpeed(id)

    @cache_results
    def vehicle_get_waiting_time(self, id):
        return traci.vehicle.getWaitingTime(id)

    @cache_results
    def tl_set_phase(self, tl_id, phase):
        self._manage_phase_durations(tl_id, phase)
        return traci.trafficlight.setPhase(tl_id, phase)

    @cache_results
    def vehicle_get_lane_position(self, id):
        return traci.vehicle.getLanePosition(id)

    def close(self):
        traci.close()

    def start(self, cmd):
        traci.start(cmd)

    @cache_results
    def get_length(self, object_type, id):
        type_to_func = {
            "lane": traci.lane.getLength
        }
        return type_to_func[object_type](id)

    @cache_results
    def vehicle_get_speed(self, id):
        return traci.vehicle.getSpeed(id)

    def simulation_step(self):
        traci.simulationStep()
        SumoBackend._CACHE = {}

    @cache_results
    def get_position(self, object_type, id):
        type_to_method = {"vehicle": traci.vehicle.getPosition,
                          "junction": traci.junction.getPosition}
        return type_to_method[object_type](id)

    @cache_results
    def lane_get_max_speed(self, id):
        return traci.lane.getMaxSpeed(id)

