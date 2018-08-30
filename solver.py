from HParams import global_hparams as gh
import numpy as np
from tqdm import trange
from collections import defaultdict, OrderedDict


class Solver():
    def __init__(self, sim_runner, logger):
        self.sim_runner = sim_runner
        # self.agents = self.sim_runner.agents
        self.junctions = self.sim_runner.junctions
        self.agents = self.sim_runner.agents
        self.num_episodes = gh.num_episodes
        self.steps_per_episode = gh.steps_per_episode
        self.observations = defaultdict()
        self.agent_to_neighbors = {a: self._get_neighbor_junctions(
            a.junction) for a in self.agents}
        self.r_weight = gh.r_weight
        self.logger = logger

        # logs formulation - {agent a: {episode: [1,1,..,num_episodes],
        # time_step: [1,2,3,...,steps_per_episode,1, ...]
        # action/rewards: [a1, a2, ...]}}

    def _calc_global_reward(self, agent, observations):
        num_neighbors = len(self.agent_to_neighbors[agent])
        n_weight = (1 - self.r_weight) / num_neighbors
        reward = observations[agent.junction].reward * self.r_weight
        for j in self.agent_to_neighbors[agent]:
            reward += observations[j].reward * n_weight
        return reward

    def _get_neighbor_junctions(self, agent_j):
        neighbors = []
        in_lanes = set(agent_j.incoming_lanes)
        for j in self.junctions:
            out_lanes = set(j.outgoing_lanes)
            if set.intersection(in_lanes, out_lanes):
                neighbors.append(j)
        return neighbors

    def _episode_step(self, episode):
        if hasattr(gh, "alpha_decay"):
            gh.alpha = gh.alpha * gh.alpha_decay
        for i in trange(self.steps_per_episode, desc="step", position=2):
            if self.logger is not None:
                self.logger["episode"] = episode
                self.logger["step"] = i
            # if it's the first step and no observations yet
            if not self.observations:
                action_per_agent = {}
            else:
                action_per_agent = {agent: agent.getAction(self.observations[
                                       agent.junction].state_action_representations)
                                    for agent in self.sim_runner.agents}
            # get new set of observations
            new_observations = self.sim_runner.run_step(action_per_agent)

            if self.observations:
                for agent in self.agents:
                    prev_observation = self.observations[agent.junction]
                    new_observation = new_observations[agent.junction]
                    action = action_per_agent[agent]
                    global_reward = self._calc_global_reward(agent,
                                                             new_observations)
                    agent.update(prev_observation, action, new_observation, global_reward)

                    # log actions and global reward
                    if self.logger is not None:
                        self.logger["a_"+agent.junction.junction_id+"_action"] = action
                        self.logger["a_"+agent.junction.junction_id+"_global_reward"] = global_reward
            if self.logger is not None:
                self.logger.dump_row()
            # update observations
            self.observations = new_observations

    def train(self):
        for i in trange(self.num_episodes, desc="episode", position=1):
            if i % 5 == 0:
                self.logger.save_agents("weights_%d_{}" % i)
            print("\n\n")
            self._episode_step(i)
            self.sim_runner.restart_simulation()

            if self.logger is not None:
                self.logger.dump_table()
                self.logger.save_agents("weights_{}")
