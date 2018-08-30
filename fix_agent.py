from agents import Agent

class FixedAgent(Agent):

    def __init__(self, fixed_action, junction):
        super().__init__(junction)
        self.fixed_action = fixed_action

    def update(self, prev_observation, action, new_observation, global_reward):
        return 0

    def getValue(self, state):
        return 0

    def get_weights(self):
        return 0

    def getAction(self, state):
        return self.fixed_action