"""
FILE: logger_reader.py
Writer: Netai Benaim
AI_project
DESCRIPTION: 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
import os
import seaborn as sns
import csv
import sys
import json




# REWARD_COMPONENTS = ["j_{}_delay", "j_{}_wait", "j_{}_emergency_brake"]
# components to plot
REWARD_COMPONENTS = ["j_{}_delay", "j_{}_wait"]
fig_num = 1
REZ = 100
SIZE = (12, 8)


def clean_item(item):
    if item == "":
        return "0"
    return item


def read_fields(path, fields, kind=float):
    """
    reads the specified fields from a csv file
    :param path: a path of the csv file
    :param fields: a list of the names of columns to return from the csv file
    :param kind: a function to cast the string from the csv, default float
    :return: an np.array where the columns are the desired columns of the csv file
    """
    lines = list()
    with open(path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            line_raw = [row[field] for field in fields]

            lines.append([kind(item) for item in map(clean_item,line_raw)])

    return np.array(lines)


def action_plot(num_episodes, episode_length, actions, num_actions=2, agent="0", title="", save=True, save_path=""):
    """
    plots the action changes as a function of steps
     :param num_episodes: number of episodes
    :param episode_length: number of steps per episode
    :param actions: a vector with the actions
    :param num_actions: number of actions available to agent
    :param agent: a string representing the agent
    :param title: title for the plot
    :param save: boolean indicator of whether or not to save
    :param save_path: path to save figure
    :return: None
    """
    global fig_num
    fig_num += 1
    plt.figure(fig_num, figsize=SIZE, dpi=REZ)
    plt.plot(actions + 1, linewidth=0.2)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("steps")
    plt.ylabel("action")
    plt.yticks([i + 1 for i in range(num_actions)], ["action_" + str(i) for i in range(num_actions)])
    plt.title(title)

    if save:
        plt.savefig(save_path + "/agent_" + agent + "_action_per_step.jpeg")
    else:
        plt.show()


def all_agents_action_plot(base_path, configuration, save=False):
    """
    plots for each agent its actions as a function of steps
    :param base_path: the path of the data directory
    :param configuration: a dictionary of configurations
    :param save: boolean indicator of whether or not to save
    :return:
    """
    act = read_fields(base_path + "/log.csv", ["a_" + ag + "_action" for ag in configuration["junctions_with_agents"]])
    title = "\naction for agent %s as function of steps\n ${\\alpha}$=%.4f ${\epsilon}$=%.4f discount=%.4f r_weight=%.4f"
    for i, agent in enumerate(configuration["junctions_with_agents"]):
        cur_title = configuration['mode'] + title % (agent, configuration['alpha'], configuration['epsilon'], configuration['discount'], configuration['r_weight'])
        action_plot(configuration["num_episodes"], configuration["steps_per_episode"], act.T[i], save=save, agent=agent, title=cur_title, save_path=base_path)


def action_ratio_plot(num_episodes, episode_length, actions, num_actions=2, agent="0", title="", save=True,
                      stack=True, save_path=""):
    """
    plots the percentage of the time an agent took an action as a function of the number of episodes
    :param num_episodes: number of episodes
    :param episode_length: number of steps per episode
    :param actions: a vector with the actions
    :param num_actions: number of actions available to agent
    :param agent: a string representing the agent
    :param title: title for the plot
    :param save: boolean indicator of whether or not to save
    :param stack: indicator for the type of plot, if true plots stack plot
    :param save_path: path to save figure
    :return: None
    """
    global fig_num
    fig_num += 1
    plt.figure(fig_num, figsize=SIZE, dpi=REZ)

    ratios = list()
    for i in range(num_episodes):
        episode_hist = np.histogram(actions[i * episode_length: (i + 1) * episode_length], bins=num_actions,
                                    range=(0, num_actions - 1))[0]

        ratios.append(episode_hist / episode_length)
    ration_mat = np.array(ratios).T

    if stack:
        plt.stackplot(range(num_episodes), ration_mat,  labels=["action_" + str(a) for a in range(num_actions)],
                      baseline="zero")

    else:
        for i, act in enumerate(ration_mat):
            plt.plot(act, label="action_" + str(i))

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("number of episodes")
    plt.ylabel("% of episode spent in action")
    # plt.gca().set_yticklabels(['{:,.2%}'.format(x / 100) for x in range(0, 110, 10)])
    plt.title(title)

    if save:
        plt.savefig(save_path + "/agent_" + agent + "_action_ratio_per_episode.jpeg")
    else:
        plt.show()


def all_agents_action_ration(base_path, configuration, stack=True, save=False):
    """
    plots the statistics about the actions for all the agents
    :param base_path: the path of the data directory
    :param configuration: a dictionary of configurations
    :param save: boolean indicator of whether or not to save
    :param stack: indicator for the type of plot, if true plots stack plot
    :return:
    """
    # act = dict()
    act = read_fields(base_path + "/log.csv", ["a_" + ag + "_action" for ag in configuration["junctions_with_agents"]])
    title = "\naction for agent %s \n ${\\alpha}$=%.4f ${\epsilon}$=%.4f discount=%.4f r_weight=%.4f"
    for i, agent in enumerate(configuration["junctions_with_agents"]):
        cur_title = configuration['mode'] + title % (agent, configuration['alpha'], configuration['epsilon'], configuration['discount'], configuration['r_weight'])
        action_ratio_plot(configuration["num_episodes"], configuration["steps_per_episode"], act.T[i], num_actions=8,
                          agent=agent, title=cur_title, save=save, stack=stack, save_path=base_path)
        # TODO: make num_actions generic


def load_config(path):
    """
    loads the configuartion json
    :param path: path of the log directory
    :return: configuration dictionary
    """
    with open(path + "/configuration.json", "r") as fd:
        return json.load(fd)


def reward_global(num_episodes, episode_length, rewards, agent="0", title="", win_size=200, save=True, save_path=""):
    """
    plots the average reward (in a window) against the actual reward
    :param num_episodes: number of episodes
    :param episode_length: number of steps in episode
    :param rewards: a vector of rewards
    :param agent: the id of the agent
    :param title: title for the plot
    :param win_size: window size for moving average
    :param save: boolean indicator of whether or not to save
    :param save_path: path to save figure
    :return None
    """
    global fig_num
    fig_num += 1
    plt.figure(fig_num, figsize=SIZE, dpi=REZ)

    # plt.suptitle("Test\n blala")
    plt.plot(rewards, label="real signal")
    plt.plot(np.convolve(rewards, np.ones((win_size,)) / win_size), label="smoothed (win size = {})".format(win_size))
    plt.legend(loc=3)
    plt.xlabel("training steps")
    plt.ylabel("global reward ")
    # plt.gca().set_yticklabels(['{:,.2%}'.format(x / 100) for x in range(0, 110, 10)])
    plt.title(title)

    if save:
        plt.savefig(save_path + "/agent_" + agent + "_global_reward_per_episode.jpeg")
    else:
        plt.show()


def reward_components(num_episodes, episode_length, rewards, agent="0", title="", save=True, save_path="", stack=False):
    """
    plots the different components of the reward
     :param num_episodes: number of episodes
    :param episode_length: number of steps in episode
    :param rewards: a vector of rewards
    :param agent: the id of the agent
    :param title: title for the plot
    :param win_size: window size for moving average
    :param save: boolean indicator of whether or not to save
    :param save_path: path to save figure
    :param stack: indicator for the type of plot, if true plots stack plot
    :return: None
    """
    global fig_num
    fig_num += 1
    plt.figure(fig_num, figsize=SIZE, dpi=REZ)

    labels = [com.format(agent) for com in REWARD_COMPONENTS]

    if stack:
        plt.stackplot(rewards, labels=labels, baseline="zero")

    else:
        for i, comp in enumerate(rewards):
            plt.plot(comp, label=labels[i])

    plt.legend(loc=3)
    plt.xlabel("training steps")
    plt.ylabel("reward value")
    plt.title(title)

    if save:
        plt.savefig(save_path + "/agent_" + agent + "_reward_components.jpeg")
    else:
        plt.show()


def reward_per_episode(num_episodes, episode_length, rewards, agent="0", title="", save=True, save_path=""):
    global fig_num
    fig_num += 1
    plt.figure(fig_num, figsize=SIZE, dpi=REZ)
    rewards_avg = list()
    for e in range(num_episodes):
        rewards_avg.append(np.mean(rewards[e * episode_length: (e + 1) * episode_length]))

    # print(rewards_avg)
    plt.plot([i for i in range(num_episodes)],rewards_avg)

    # plt.legend(loc=3)
    plt.xlabel("episodes")
    plt.ylabel("average reward value")
    plt.title(title)

    if save:
        plt.savefig(save_path + "/agent_" + agent + "_avg_reward.jpeg")
    else:
        plt.show()


def all_agents_reward(base_path, configuration, component_plot="junctions", stack=True, save=False):
    """
    plots the statistics about the rewards for all the agents
    :param base_path: the path of the data directory
    :param configuration: a dictionary of configurations
    :param component_plot: "junctions" to plot all junctions or
                            "junctions_with_agents" to plot agents' junctions
    :param stack: indicator for the type of plot, if true plots stack plot
    :return:
    """

    title = "\nfor agent %s \n ${\\alpha}$=%.4f ${\epsilon}$=%.4f discount=%.4f r_weight=%.4f"
    rews = read_fields(base_path + "/log.csv", ["a_" + ag + "_global_reward" for ag in configuration["junctions_with_agents"]])

    for i, agent in enumerate(configuration["junctions_with_agents"]):
        cur_title = configuration['mode'] + "global reward" + title % (agent, configuration['alpha'], configuration['epsilon'], configuration['discount'], configuration['r_weight'])
        reward_global(configuration["num_episodes"], configuration["steps_per_episode"], rews.T[i],
                          agent=agent, title=cur_title, save=save, save_path=base_path)
        cur_title = configuration['mode'] + "avg reward" + title % (agent, configuration['alpha'], configuration['epsilon'], configuration['discount'], configuration['r_weight'])
        reward_per_episode(configuration["num_episodes"], configuration["steps_per_episode"], rews.T[i],
                           agent=agent, title=cur_title, save=save, save_path=base_path)

        title = configuration['mode'] + "\nreward components for junction %s \n ${\\alpha}$=%.4f ${\epsilon}$=%.4f discount=%.4f r_weight=%.4f"
    # for junc in configuration[component_plot]:
    #     cur_title = title % (
    #     junc, configuration['alpha'], configuration['epsilon'], configuration['discount'], configuration['r_weight'])
    #     rew_comp = read_fields(base_path + "/log.csv", [com.format(junc) for com in REWARD_COMPONENTS])
    #     reward_components(configuration["num_episodes"], configuration["steps_per_episode"], rew_comp.T,
    #                       agent=junc, title=cur_title, save=save, save_path=base_path, stack=False)


def calculate_average_episode_value(log , keys, name=None):
    if name in ["LQF", "AUTOMATIC"]:
        per_episode_means = [pd.DataFrame({"episode":range(150), PARAM_NAME: log.groupby(
            "episode")[key].mean().mean()})  for key in keys]
    else:
        per_episode_means = [pd.DataFrame({"episode": log["episode"].unique(), PARAM_NAME: log.groupby(
            "episode")[key].mean()}) for key in keys]
    return {key: means for key, means in zip(keys, per_episode_means)}


def calculate_average_episode_reward(log, name=None):
    keys = log.keys()
    global_r_keys = []
    for k in keys:
        if PARAM in k:
            for j in JUNCTIONS:
                if j in k:
                    global_r_keys.append(k)
                    break
    # global_r_keys = list(filter(lambda k: PARAM in k and any([k.startswith(j) for j in JUNCTIONS])  in
    #                                                      JUNCTIONS,
    #                                                          log.keys()))
    return calculate_average_episode_value(log, global_r_keys, name)


def compare_runs_episode_reward(all_paths):
    """
    plots the global rewards mean per episode of agents from different experiments.
    :param all_paths: list of paths for log directories
    """
    logs = [pd.read_csv(os.path.join(path, "log.csv")) for path in all_paths]
    average_rewards = [calculate_average_episode_reward(log, name) for log, name in zip(logs, all_names)]
    all_datapoints = []
    # arrange data in tidy format
    for rewards, path, name in zip(average_rewards, all_paths, all_names):
        for agent_key, log in rewards.items():
            log["agent"] = agent_key.rstrip("_" + PARAM)
            log["experiment"] = name
            all_datapoints.append(log)
    data = pd.concat(all_datapoints)
    # sns.relplot(x='episode', y=PARAM_NAME, hue='experiment', data=data,
    #             kind="line")
    sns.relplot(x='episode', y=PARAM_NAME, hue='experiment', style='agent', data=data,
                kind="line")
    # plt.savefig("two_agents_reward_episodes.svg")
    plt.savefig("two_agents_reward_episodes.pdf")
    # plt.savefig("two_agents_reward_episodes.jpg")
    # plt.show()

def compare_step_rewards(all_paths, max_steps=2000, window_size=30):
    """
    plots the global rewards per step (smoothed) of agents from different experiments.
    :param all_paths: list of paths for log directories
    :param max_steps: maximal number of steps to show
    :param window_size: rolling window for smoothing
    """
    logs = [pd.read_csv(os.path.join(path, "log.csv"))[:max_steps] for path in all_paths]
    all_data = []
    for path, log in zip(all_paths, logs):
        log["step"] = range(0, len(log)) # unite the steps count from all episodes
        r_keys = list(filter(lambda k: PARAM in k and k[:4] in JUNCTIONS, log.keys()))
        for key in r_keys:
            # generate data in tidy format for specific agent in specific experiment
            agent_data = pd.DataFrame({"step": log["step"],
                                       "agent": key.rstrip("_wait"),
                                       "experiment": path.split("/")[-1],
                                       PARAM_NAME: log[key].rolling(window=window_size,
                                                                     center=False).mean()})
            all_data.append(agent_data)
    data = pd.concat(all_data)
    sns.relplot(x='step', y=PARAM_NAME, hue='experiment', style='agent', data=data,
                kind="line")
    # plt.semilogy()

    plt.show()


def compare_test_averages(all_paths):
    logs = [pd.read_csv(os.path.join(path, "log.csv")) for path in all_paths]
    average_rewards = [calculate_average_episode_reward(log) for log in logs]
    all_datapoints = []
    # arrange data in tidy format
    for rewards, path, name in zip(average_rewards, all_paths, all_names):
        for agent_key, log in rewards.items():
            log["agent"] = agent_key.rstrip("_" + PARAM)
            log["experiment"] = name
            all_datapoints.append(log)
    data = pd.concat(all_datapoints)
    sns.barplot(x='agent', y=PARAM_NAME, hue='experiment', data=data, capsize=0.2)
    # plt.savefig("box_two_agents_teleports_average.svg")
    plt.savefig("bar_two_agents_teleports_average.pdf")
    # plt.savefig("box_two_agents_teleports_average.jpg")


def compare_total_reward_averages():
    logs = [pd.read_csv(os.path.join(path, "log.csv")) for path in all_paths]
    # all_keys = logs[0].keys()
    # param_keys = list(filter(lambda k: PARAM in k, all_keys))

    average_rewards = [calculate_average_episode_reward(log) for log in logs]
    all_datapoints = []
    # arrange data in tidy format
    for rewards, path, name in zip(average_rewards, all_paths, all_names):
        for agent_key, log in rewards.items():
            log["agent"] = agent_key.rstrip("_" + PARAM)
            log["experiment"] = name
            all_datapoints.append(log)
    data = pd.concat(all_datapoints)
    sns.boxplot(y=PARAM_NAME, x='experiment', data=data)
    # plt.show()
    plt.savefig("two_agents_total_reward_all.svg")
    plt.savefig("two_agents_total_reward_all.jpg")
    plt.savefig("two_agents_total_reward_all.pdf")


if __name__ == '__main__':
    # if len(sys.argv) < 2:
    #     print("usage: python logger_reader.py <path to log dir>")
    #     exit(1)
    #
    # base_path = sys.argv[1]
    # all_paths = sys.argv[1:]
    # JUNCTIONS = ["j_0_", "j_1_", "j_0d_", "j_0l_", "j_0u_", "j_1u_", "j_1r_", "j_1d_"]
    JUNCTIONS = ["j_0_", "j_1_"]
    # JUNCTIONS = ["j_0_"]
    PARAM  = "teleport"
    PARAM_NAME = "average number of collisions per step"
    PARAMS = ["teleport", "num_cars", "total_w"]
    PARAMS_NAME = [
        "average number of teleports per step",
        "average number of cars in junction per step",
        "average local reward per step"
        ]
    all_paths = [
        "/Users/danamir/CS/AI/AI_project/logs/230818053414_straight_p0.02_two_decay_test",
        "/Users/danamir/CS/AI/AI_project/logs/220818172004_220818_test_p002_two_decay_lqf",
        "/Users/danamir/CS/AI/AI_project/logs/220818173811_220818_test_p002_two_decay_auto"
    ]
    all_names = ["RL", "LQF" , "AUTOMATIC"]
    base_path = all_paths[0]
    configuration = load_config(base_path)
    
    # all_agents_action_ration(base_path, configuration, stack=True, save=False)
    # all_agents_action_plot(base_path, configuration, save=False)
    # all_agents_reward(base_path, configuration, save=False)
    # all_agents_reward(base_path, configuration, save=True, stack=True)
    # compare_runs_episode_reward(all_paths)
    # compare_step_rewards(all_paths)
    compare_test_averages(all_paths)
    # compare_total_reward_averages()