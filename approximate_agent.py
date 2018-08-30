from agents import Agent
import numpy as np
import random
from HParams import global_hparams as gh


class Approximate(Agent):

    def __init__(self, discount, alpha, epsilon, junction, weights=None):
        super().__init__(junction)
        self.discount = discount
        self.alpha = alpha
        self.weights = weights
        # self.junction = junction
        self.epsilon = epsilon

    def getQValue(self, features):
        """
        computes the Q value of the feature representing a state
        :param features: ndarray of features
        :return: Q value of the states
        """
        if self.weights is None:
            self.weights = np.zeros(features.shape)
        return np.inner(features, self.weights)

    def getAction(self, actions_features):
        """
        returns the best action to do in the current state
        :param actions_features: dictionary of actions as keys and their feature vectors
        as values
        :return: an action to preform, None if no legal actions are available
        """
        legal_actions = list(actions_features.keys())
        if len(legal_actions) == 0:
            return None

        if np.random.binomial(1, self.epsilon, 1):
            return random.choice(legal_actions)

        else:
            qs = np.array([self.getQValue(actions_features[a]) for a in legal_actions])
            max_val = self.getValue(actions_features)
            max_indices = np.nonzero(np.abs(qs - max_val) < 1e-6)
            return legal_actions[random.choice(max_indices[0])]

    def getValue(self, actions_features):
        """
        computes max_action Q(state,action)
        :param actions_features: dictionary of actions as keys and their feature vectors
        as values
        :return: max_action Q(state,action)
        """
        if len(actions_features) == 0:
            return - np.inf
        v = max([self.getQValue(feat) for feat in actions_features.values()])
        return v

    def update(self, prev_observation, action, new_observation, global_reward):
        """
        updates the weights of the vector
        :param prev_observation: previous state and observation
        :param action: action taken
        :param new_observation: next state
        :param global_reward: reward from manager
        """
        f = prev_observation.state_action_representations[action]
        correction = (global_reward + (self.discount *
                    self.getValue(new_observation.state_action_representations)) -
                      self.getQValue(f)) * gh.alpha
        self.weights = self.weights + correction * f

    def __hash__(self):
        return self.junction.__hash__()

    def get_weights(self):
        return self.weights
