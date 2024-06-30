# causal_inference.py
from sklearn.linear_model import LogisticRegression
import numpy as np

class CausalInference:
    def __init__(self):
        self.propensity_model = LogisticRegression()

    def fit_propensity_model(self, states, actions):
        self.propensity_model.fit(states, actions)

    def predict_propensity(self, state):
        return self.propensity_model.predict_proba(state)[:, 1]

    def adjust_rewards(self, states, actions, rewards):
        propensities = self.predict_propensity(states)
        adjusted_rewards = rewards / propensities
        return adjusted_rewards
