# causal_bid_agent.py
import numpy as np
from bid_agent import BidAgent  # Ensure to import the base agent class
from causal_inference import CausalInference
from data_collector import DataCollector

class CausalBidAgent(BidAgent):
    def __init__(self, cfg, state_size, cat_sizes=tuple(), test_mode=False, envname='yewu', writer=None):
        super().__init__(cfg, state_size, cat_sizes, test_mode, envname, writer)
        self.causal_inference = CausalInference()
        self.data_collector = DataCollector()

    def fit_propensity_model(self):
        states, actions, rewards, next_states, dones = self.data_collector.get_data()
        self.causal_inference.fit_propensity_model(states, actions)

    def adjust_rewards(self):
        states, actions, rewards, next_states, dones = self.data_collector.get_data()
        adjusted_rewards = self.causal_inference.adjust_rewards(states, actions, rewards)
        return states, actions, adjusted_rewards, next_states, dones

    def act(self, state):
        if self.ratio_update_type == 1:
            last_ratio = state[1]
            allowed_idxes = [last_ratio + x > 0 for x in self._get_action_space()]
        else:
            allowed_idxes = None

        action_next_slot = self.agent.act(state, eps=self.eps if self.during_exp() else 1.0, is_deterministic=not self.during_exp(),
                                          allowed=allowed_idxes)
        return action_next_slot

    def learn(self, can_train=False, iter_multiplier=1):
        if can_train:
            # Adjust rewards using causal inference
            self.fit_propensity_model()
            states, actions, adjusted_rewards, next_states, dones = self.adjust_rewards()

            for i in range(len(states)):
                self.update_buffer((states[i], actions[i], adjusted_rewards[i], next_states[i], dones[i]))

            # Proceed with training as usual
            super().learn(can_train, iter_multiplier)
            
    def batch_act(self, obs):
        action_next_slot = self.agent.act(obs, eps=self.eps if self.during_exp() else 1.0, is_deterministic=not self.during_exp())
        return action_next_slot

    def evaluate_policy(self):
        data = self.collect_evaluation_data()
        self.fit_propensity_model(data)
        total_reward = 0

        for state, action, reward in data:
            propensity = self.causal_inference.predict_propensity(state)
            adjusted_reward = reward / propensity
            total_reward += adjusted_reward

        average_reward = total_reward / len(data)
        return average_reward

    def collect_data(self, state, action, reward, next_state, done):
        self.data_collector.collect(state, action, reward, next_state, done)
