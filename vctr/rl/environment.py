import gym
from gym import spaces
import numpy as np


class TradingEnvironment(gym.Env):
    def __init__(self, data):
        super(TradingEnvironment, self).__init__()
        self.data = data
        self.current_step = 0

        # Define the action space
        self.action_space = spaces.Discrete(3)  # hold/buy/sell

        # Define the observation space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(data.columns),), dtype=np.float32)

        self.current_position = None
        self.balance = 10000  # Initial account balance
        self.trade_fee = 10  # Trade fee for each buy or sell action
        self.holding_penalty = 0.01  # Penalty for holding without an open position

    def step(self, action):
        # Update the environment based on the action taken
        self.current_step += 1

        observation = self.data.iloc[self.current_step]
        reward = self.calculate_reward(action)
        self.update_position_and_balance(action)
        done = self.current_step >= len(self.data) - 1

        # Todo: Implement this properly.
        truncated = False
        return observation, reward, done, truncated, {}

    def reset(self):
        self.current_step = 0
        return self.data.iloc[self.current_step]

    def calculate_reward(self, action):
        current_price = self.data.iloc[self.current_step]['close']
        next_price = self.data.iloc[self.current_step + 1]['close']

        if action != 0 and action == 1 and self.current_position is None:
            reward = next_price - current_price - self.trade_fee
        elif (
            action != 0
            and action == 1
            or action != 0
            and action == 2
            and self.current_position is None
            or action == 0
        ):
            reward = 0  # Already holding a position
        elif action == 2:
            reward = current_price - self.current_position - self.trade_fee
            self.current_position = None

        # Normalize the reward by dividing by the current price
        reward /= current_price

        return reward

    def update_position_and_balance(self, action):
        current_price = self.data.iloc[self.current_step]['close']

        if action == 1 and self.current_position is None:  # buy
            self.current_position = current_price
            self.balance -= current_price + self.trade_fee
        elif action == 2 and self.current_position is not None:  # sell
            self.balance += current_price - self.trade_fee
            self.current_position = None
