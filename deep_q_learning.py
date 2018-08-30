from agents import Agent
import numpy as np
import random


class DeepQLearner(Agent):
    def __init__(self, junction, discount, alpha, epsilon, model, num_actions=2, batch_size=50):
        super().__init__(junction)
        self.discount = discount
        self.alpha = alpha
        self.epsilon = epsilon
        self.model = model
        self.batch_size = batch_size
        self.legal_actions = [i for i in range(num_actions)]
        self.obseravations = list()

    def update(self, prev_observation, action, new_observation, global_reward):

        self.obseravations.append((prev_observation, action, global_reward, new_observation))

        if len(self.obseravations) == self.batch_size:
            self._train()
            self.obseravations = list()

    def _train(self):
        inputs_shape = (self.batch_size,) + self.obseravations[0][0].shape
        inputs = np.zeros(inputs_shape)
        targets = np.zeros((self.batch_size, len(self.legal_actions)))

        for i in range(self.batch_size):
            state = self.obseravations[i][0]
            action = self.obseravations[i][1]
            reward = self.obseravations[i][2]
            state_new = self.obseravations[i][3]

            # Build Bellman equation for the Q function
            inputs[i:i + 1] = np.expand_dims(state, axis=0)
            targets[i] = self.model.predict(state)
            Q_sa = self.model.predict(state_new)

            targets[i, action] = reward + self.discount * np.max(Q_sa)
            self.model.train_on_batch(inputs, targets)

    def get_weights(self):
        return self.model

    def getValue(self, actions_features):
        Q = self.model.predict(actions_features)
        return np.max(Q)

    def getAction(self, actions_features):

        if np.random.binomial(1, self.epsilon, 1):
            return random.choice(self.legal_actions)

        Q = self.model.predict(actions_features)
        return np.argmax(Q)