from logging import log
from HParams import global_hparams as gh
from collections import defaultdict
from io_utils import filename_from_path

import shutil
import csv
import json
import os
import time

# visualization
# we don't have a display
# import matplotlib as mpl
#
# mpl.use('Agg')
# import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns

class ModelLogger():
    def __init__(self, agents, config, fixed_keys):
        """

        :param agents: list of the agents
        :param config: path to the configuration file used to run this experiment
        :param fixed_keys: list of the keys that will be used in the logging
        """
        # get the correct identifier and create a dir for the experiment
        self.identifier = self._get_identifier(filename_from_path(config))

        self.base_dir = os.path.join(gh.log_path, self.identifier)

        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        shutil.copy(config, os.path.join(self.base_dir, "configuration.json"))

        self.config = config
        self.fixed_keys = fixed_keys
        self.row_buffer = {key: "" for key in self.fixed_keys}
        self.table_buffer = []
        self.log_path = os.path.join(self.base_dir, "log.csv")

        assert all([type(key) == str for key in fixed_keys])
        with open(self.log_path, "w") as log_file:
            log_file.write(",".join(self.fixed_keys)+"\n")

        self.agents = agents
        self.last_model = None

    def __setitem__(self, key, value):
        """
        sets the value of a log key in the current row.
        :param key: valid key from self.fixed_keys
        :param value: any valid value
        """
        if key not in self.row_buffer:
            raise ValueError("invalid key for logger - {}".format(key))
        self.row_buffer[key] = value

    def dump_row(self):
        """
        dump the current row to the table and starts a new empty row
        """
        self.table_buffer.append(self.row_buffer.copy())
        self.row_buffer = {key: "" for key in self.fixed_keys}

    def dump_table(self):
        """
        appends the current table to the log file.
        """
        with open(self.log_path, "a") as f:
            for row in self.table_buffer:
                line = ",".join([str(row[key]) for key in self.fixed_keys])+"\n"
                f.write(line)
        self.table_buffer = []

    def save_agents(self, name):
        """
        saves the weights of all agents to numpy files in the log dir
        :param name: a name for the file that can be formatted with the ID of the agent.
        """
        for agent in self.agents:
            try:
                weights = agent.get_weights()
                agent_id = agent.junction.junction_id
                path = os.path.join(self.base_dir, name.format(agent_id))
                np.save(path, weights)
            except NotImplementedError:
                pass


    @staticmethod
    def _get_identifier(mode, resume='new_run'):
        t_format = '%d%m%y%H%M%S'
        # new training instance
        if resume == 'new_run':
            time_string = time.strftime(t_format)
            identifier = '{}_{}'.format(time_string, mode)

        # last known try
        else:
            identifier = resume

        return identifier

