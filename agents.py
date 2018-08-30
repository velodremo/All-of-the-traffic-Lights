
class Agent:
    def __init__(self, junction):
        self.junction = junction

    def getAction(self, state):
        raise NotImplementedError()

    def getValue(self, state):
        raise NotImplementedError()

    def update(self, prev_observation, action, new_observation, global_reward):
        raise NotImplementedError()

    def get_weights(self):
        raise NotImplementedError()
