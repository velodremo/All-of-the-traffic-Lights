# from agents import Agent


class Explorer():
    def __init__(self, agent, program):
        """

        :param agent: an agent with epsilon greedy
        :param program: a list of tuples (num_steps, epsilon), first cell number of steps to use this epsilon
        """
        # super().__init__(junction)
        self.program = program
        self.agent = agent
        self.junction = self.agent.junction
        self.iters = 0

    def getValue(self, state):
        self.agent.getValue(state)

    def getAction(self, actions_features):
        if len(self.program) != 0:
            if self.iters == self.program[0][0]:
                next_prob = self.program.pop()
                self.agent.epsilon = next_prob[1]
                self.iters = 0
        else:
            self.agent.epsilon = 0

        self.iters += 1
        return self.agent.getAction(actions_features)

    def update(self, prev_observation, action, new_observation, global_reward):
        self.agent.update(prev_observation, action, new_observation, global_reward)

    def get_weights(self):
        return self.agent.get_weights()