import numpy as np
import logging
from tqdm import tqdm
from HParams import global_hparams as gh

class Observation(object):
    """
    a value class which represents an observation in the MDP
    """
    def __init__(self, reward, state_action_representations):
        """

        :param reward: single float number
        :param state_action_representations: dictionary of mappings from actions to state-action
        representation
        """
        self.reward = reward
        self.state_action_representations = state_action_representations


class SimulationRunner(object):
    """
    An object used for communicationg with the simulation according to the agents behavior:
    in each step the runner:
    -executes the given actions of agents
    -returns the observations of agents
    -returns the partial observations (only rewards) of non-agent junctions
    """
    def __init__(self, agents, junctions, sumo_cmd, backend, logger):
        """
        :param agents: list of Agent objects that represents the traffic lights agents
        :param junctions: list of junction objects of all the junctions that are observed
        :param sumo_cmd: the command used for starting the simulation
        :param backend: a sumo backend object
        """
        self.sumo_cmd = sumo_cmd
        self.agents = agents
        self.junctions = junctions
        self.j_to_a = {a.junction: a for a in agents}
        self.timestep = 0
        self.state_extractors = None
        self.reward_extractors = None
        self.backend = backend
        self.logger = logger

    def close_simulation(self):
        """
        shuts down sumo
        """
        self.backend.close()

    def restart_simulation(self):
        """
        restarts the simulation
        """
        self.backend.close()
        self.backend.start(self.sumo_cmd)
        for r_extractor in self.reward_extractors.values():
            r_extractor.restart()
        if hasattr(gh, "simulation_offset"):
            for i in range(gh.simulation_offset):
                self.backend.simulation_step()


    def get_allowed_actions(self, agent):
        """
        abstract method - used to define the list of possible actions for a given agent.
        """
        raise NotImplementedError()

    def run_step(self, action_per_agent):
        """
        run a single step of simulation. executes the given actions of all given agents and returns
        observations for all junctions. Observations of non-agent junctions will include only rewards.
        :param action_per_agent: dictionary of agent -> action
        :return: dictionary of junction -> observation
        """
        for agent, action in action_per_agent.items():
            self._perform_action(agent, action)
        self.backend.simulation_step()
        observation_per_junction = {}
        vehicle_list = self.backend.get_id_list("vehicle")
        vehicles_per_junction = self._filter_vehicles_per_junction(vehicle_list)
        for junction in self.junctions:
            vl = vehicles_per_junction[junction]
            sa_reps = {}
            if junction in self.j_to_a:
                agent = self.j_to_a[junction]
                se = self.state_extractors[agent]
                for action in self.get_allowed_actions(agent):
                    sa_reps[action] = se.extract_state(vl, action)
            re = self.reward_extractors[junction]
            reward = re.get_reward(vl)
            observation = Observation(reward=reward, state_action_representations=sa_reps)
            observation_per_junction[junction] = observation
        self.timestep += 1
        return observation_per_junction

    def _filter_vehicles_per_junction(self, vehicle_list):
        # generates dictionary of seperate vehicle lists for the different junctions
        vehicles_per_junction = {junction: [] for junction in self.junctions}
        if len(vehicle_list) == 0:
            return vehicles_per_junction

        j_locations = np.array([self.backend.get_position("junction", j.junction_id) for j in
                                self.junctions])[np.newaxis, ...]
        car_locations = np.array([self.backend.get_position("vehicle", vid) for vid in vehicle_list])[
            np.newaxis, ...]
        j_locations = np.transpose(j_locations, [1, 0, 2])
        assert j_locations.ndim == car_locations.ndim
        assert j_locations.shape[2] == car_locations.shape[2]
        distances = np.linalg.norm(j_locations - car_locations, axis=2)
        for junction, j_distances in zip(self.junctions, distances):
            v_dists = filter(lambda vd: vd[1] < gh.OBSERVED_RADIUS,list(zip(vehicle_list, j_distances)))
            vehicles_per_junction[junction] = [v_dist[0] for v_dist in v_dists]

        return vehicles_per_junction

    def _perform_action(self, agent, action):
        """
        abstract method - used to perform a specific agent of a specific agent
        :param agent: the agent who performs the action
        :param action: the action to perform (one from the list of allowed actions)
        """
        raise NotImplementedError()

    def _generate_epsilon_program(self, num_steps, num_episodes,
                                  init_epsilon, split,
                                decay=0.5):
        """

        :param num_steps: steps per episode
        :param num_episodes: number of episodes
        :param init_epsilon: initial epsilon value to be decayed
        :param split: number of splits (i.e. decays) >= 1
        :param decay: decay factor (default)
        :return: a list of (num_steps, epsilon) tuples
        """
        iters = num_steps * num_episodes
        split_len = int(iters / split)
        return [(split_len, init_epsilon*(decay**i)) for i in range(split)]


class SimulationRunnerPhaseAction(SimulationRunner):
    """
    runs the simulation with agent that can choose any phase at any step
    """
    def __init__(self, agents, junctions, sumo_cmd, state_extractor_cls, reward_extractor_class, backend,
                 logger):
        super().__init__(agents, junctions, sumo_cmd, backend, logger)
        self.state_extractors = {a: state_extractor_cls(a.junction, gh.num_phases, gh.num_phases,
                                                        backend, logger) for  a in self.agents}
        self.reward_extractors = {j: reward_extractor_class(j, backend, logger) for j in junctions}

    def get_allowed_actions(self, agent):
        if hasattr(gh, "min_phase_duration") and \
                        self.backend.tl_get_phase_passed_duration(agent.junction.traffic_light_id) < gh.min_phase_duration:
            actions =  [self.backend.tl_get_phase(agent.junction.traffic_light_id)]
        else:

            actions = list(range(gh.num_phases))
        return actions

    def _perform_action(self, agent, action):
        self.backend.tl_set_phase(agent.junction.traffic_light_id, action)


class SimulationRunnerDualAction(SimulationRunner):
    """
    runs the simulation with agent that can choose at any step whether to stay in the same phase or move to
    the next.
    """
    def __init__(self, agents, junctions, sumo_cmd, state_extractor_cls, reward_extractor_class, backend,
                 logger):
        super().__init__(agents, junctions, sumo_cmd, backend, logger)
        self.state_extractors = {a: state_extractor_cls(a.junction, 2, gh.num_phases, backend, logger) for
                                 a in
                                 self.agents}
        self.reward_extractors = {j: reward_extractor_class(j, backend, logger) for j in junctions}

    def _perform_action(self, agent, action):
        tl_id = agent.junction.traffic_light_id
        cur_phase = self.backend.tl_get_phase(tl_id)
        if action == 1:
            new_phase = (cur_phase + 1) % gh.num_phases
        else:
            new_phase = cur_phase
        self.backend.tl_set_phase(agent.junction.traffic_light_id, new_phase)

    def get_allowed_actions(self, agent):
        if hasattr(gh, "min_phase_duration") and \
                        self.backend.tl_get_phase_passed_duration(
                            agent.junction.traffic_light_id) < gh.min_phase_duration:
            actions = [0]
        else:
            actions = list(range(2))
        return actions


class SimulationRunnerAutomatic(SimulationRunner):

    def __init__(self, agents, junctions, sumo_cmd, state_extractor_cls, reward_extractor_class, backend, logger):
        super().__init__(agents, junctions, sumo_cmd, backend, logger)

        self.state_extractors = {a: state_extractor_cls(a.junction, gh.num_phases, gh.num_phases, backend,
                                                        logger
                                                        ) for a in
                                 self.agents}
        self.reward_extractors = {j: reward_extractor_class(j, backend, logger) for j in junctions}

    def get_allowed_actions(self, agent):
        return list(range(gh.num_phases))

    def _perform_action(self, agent, action):
        pass
