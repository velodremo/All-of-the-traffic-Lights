import libsumo
from dummy_agent import DummyAgent
from simulation_runner import SimulationRunner
from reward_extractor import RewardExtractor
from state_representation import StateExtractor

SUMO_CMD = ["-C", "/AI_project/nets/complex_net/config.sumocfg"]


# docker command example:
#       docker run --name <container_name> --rm -it -v <LOCAL_PATH_TO_PROJECT>/AI_project:/AI_project
# <sumo_image>:<tag> bash
#

if __name__ == '__main__':
    libsumo.start(SUMO_CMD)
    agents = [DummyAgent("0"), DummyAgent("1")]
    sr = SimulationRunner(agents, SUMO_CMD, StateExtractor, RewardExtractor)
    sr.initialize_simulation()
    for i in range(1000):
        observations = sr.run_step([agent.get_action() for agent in agents])
        print(observations)
