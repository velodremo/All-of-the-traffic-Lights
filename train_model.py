import json
import sys
from xml.etree import ElementTree as ET
import shutil
import os
from string import Template
import numpy as np

from HParams import global_hparams as gh
from approximate_agent import Approximate
from lqf_agent import LQFAgent
from fix_agent import FixedAgent
from exploring_agent import Explorer
from state_representation import StateExtractor, StateExtractorCarCounter ,StateExtractorCompact
from reward_extractor import RewardExtractor, RewardExtractorWait
from sumo_utils import build_junctions
from simulation_runner import SimulationRunnerPhaseAction, SimulationRunnerDualAction, SimulationRunnerAutomatic
from solver import Solver
from model_logger import ModelLogger
import sumo_backend
from io_utils import safe_delete_dir
ROUTE_FILE = "traffic.rou.xml"
NET_FILE = "net.net.xml"


def get_constructors():
    runner_inits = {
        "dual": SimulationRunnerDualAction,
        "phase": SimulationRunnerPhaseAction,
        "automatic": SimulationRunnerAutomatic
    }
    sr_init = runner_inits[gh.actions_type]
    s_init = StateExtractorCompact

    if gh.mode == "linear":
        def construct_approx(junction):
            return Approximate(gh.discount, gh.alpha, gh.epsilon, junction)
        a_init = construct_approx
    elif gh.mode == "fixed":
        def fixed_constructor(junction):
            return FixedAgent(gh.fixed_action, junction)
        a_init = fixed_constructor
    elif gh.mode == "lqf":
        a_init = LQFAgent
        s_init = StateExtractorCarCounter
    else:
        raise ValueError("Invalid training mode type: %s" % gh.mode)

    if gh.exploring:
        b_init = a_init
        def exploring_constructor(junction):
            return Explorer(b_init(junction), _generate_epsilon_program())
        a_init = exploring_constructor

    if not hasattr(gh, "reward_weights"):
        r_init = RewardExtractor
    else:
        delay_w = gh.reward_weights["delay"]
        wait_w = gh.reward_weights["wait"]
        flicker_w = gh.reward_weights["flicker"]
        emergency_w = gh.reward_weights["emergency"]
        teleport_w = gh.reward_weights["teleport"]

        def r_constructor(junction, backend, logger):
            return RewardExtractor(junction,
                                   backend,
                                   logger,
                                   delay_w=delay_w,
                                   flicker_w=flicker_w,
                                   wait_w=wait_w,
                                   teleport_w=teleport_w,
                                   emergency_brake_w=emergency_w)
        r_init = r_constructor


    return a_init, r_init, s_init, sr_init

def _generate_epsilon_program(split=10, decay=0.9):
    """
    Generate a list of decayed epsilon values for exploring agent
    :param num_steps: steps per episode
    :param num_episodes: number of episodes
    :param init_epsilon: initial epsilon value to be decayed
    :param split: number of splits (i.e. decays) >= 1
    :param decay: decay factor (default)
    :return: a list of (num_steps, epsilon) tuples
    """
    iters = gh.steps_per_episode * gh.num_episodes
    init_epsilon = gh.epsilon
    split_len = int(iters / split)
    program = [(split_len, init_epsilon*(decay**i)) for i in range(1, split)]
    program = program[::-1]
    return program


def update_tl_program(tree):
    """
    sets the default traffic lights programs.
    :param tree: xml tree of net file
    :return: updated xml tree
    """
    root = tree.getroot()
    programs = gh.default_program
    setattr(gh, "num_phases", len(list(programs.values())[0]))
    for child in root.findall("tlLogic"):
        if child.get("id") in programs:
            durations = programs[child.get("id")]
            for d, p in zip(durations, child):
                p.set("duration", d)
    return tree


def prepare_net(backend):
    """
    updates the net configuration files according to the training configuration HParams.
    :param backend: sumo backend used by this run
    :return: the remporary dir with the net configurations
    """

    temp_dir = "/temp"
    safe_delete_dir(temp_dir)
    shutil.copytree(gh.net_dir, temp_dir)
    temp_route = os.path.join(temp_dir, ROUTE_FILE)
    with open(temp_route, "r") as f:
        route_data = f.read()
    os.remove(temp_route)
    parameters_update_dict = gh.flow_probabilities.copy()
    parameters_update_dict["end"] = gh.steps_per_episode
    with open(temp_route, "w") as f:
        f.write(Template(route_data).substitute(parameters_update_dict))

    temp_net = os.path.join(temp_dir, NET_FILE)
    updated_tree = update_tl_program(ET.parse(temp_net))
    os.remove(temp_net)
    updated_tree.write(temp_net)
    path_idx = [s.lower() for s in gh.sumo_cmd].index("-c") + 1
    gh.sumo_cmd[path_idx] = os.path.join(temp_dir, gh.sumo_cmd[path_idx].split("/")[-1])
    backend.start(gh.sumo_cmd)
    return temp_dir # returns the temporary dir to allow it's deletion when training ends


def generate_log_keys(agents, junctions):
    keys = ["episode", "step"]
    junction_keys = ["delay", "wait", "emergency_brake", "flicker", "total_w", "teleport", "num_cars"]
    agent_keys = ["action", "global_reward"]
    for agent in agents:
        keys.extend(["a_"+agent.junction.junction_id+"_"+a_key for a_key in agent_keys])
        keys.extend(["tl_"+ agent.junction.traffic_light_id + "_phase"])
    for junction in junctions:
        keys.extend(["j_" + junction.junction_id + "_" + j_key for j_key in junction_keys])
    return keys


def run_single_model(conf_path):
    sumo_backend.set_backend(gh.backend)
    if gh.backend == "traci":
        gh.sumo_cmd[0] = "-c"
        gh.sumo_cmd = ["sumo"] + gh.sumo_cmd
    elif gh.backend in "traci-gui":
        gh.sumo_cmd[0] = "-c"
        gh.sumo_cmd = ["sumo-gui"] + gh.sumo_cmd
    backend = sumo_backend.get_backend()
    temp_dir = prepare_net(backend)

    junctions = build_junctions()
    a_class, r_class, s_class, sr_class = get_constructors()
    agents = [a_class(j) for j in junctions if j.junction_id in
              gh.junctions_with_agents]
    if hasattr(gh, "load_weights") and gh.load_weights:
        for agent, w_file in zip(agents, gh.load_weights):
            agent.weights = np.load(w_file)
    log_keys = generate_log_keys(agents, junctions)
    logger = None
    if gh.log:
        logger = ModelLogger(agents, conf_path, log_keys)
    sim_runner = sr_class(agents=agents,
                          junctions=junctions,
                          sumo_cmd=gh.sumo_cmd,
                          state_extractor_cls=s_class,
                          reward_extractor_class=r_class,
                          backend=sumo_backend.get_backend(),
                          logger=logger)
    solver = Solver(sim_runner=sim_runner, logger=logger)
    rewards = solver.train()
    sim_runner.close_simulation()
    shutil.rmtree(temp_dir)
    return rewards


def main():
    """
    runs a training procedure using the given configuration file.
    USAGE: train_model.py <config>
    for grid search add "-g <params> where params is subset of "edns" for the following params:
    epsilon (e), discount(d), num_episodes(n), num_steps (s)
    """
    assert len(sys.argv) >= 2
    conf_path = sys.argv[1]

    with open(conf_path, 'r') as f:
        conf_dict = json.load(f)
        gh.set_params(conf_dict)

    run_single_model(conf_path)


if __name__ == '__main__':
    main()

