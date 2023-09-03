import gym
from gym import spaces
import numpy as np


class TradingEnvironment(gym.Env):
    def __init__(self, data, close_values, initial_balance=10000, transaction_cost=0.001):
        super(TradingEnvironment, self).__init__()

        self.data = data
        self.close_values = close_values

        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost

        self.action_space = spaces.Discrete(3)  # HOLD, BUY, SELL
        self.observation_space = spaces.Box(low=0, high=1, shape=(data.shape[1],), dtype=np.float32)

        self.reset()

    def step(self, action):
        # Update the current step
        self.current_step += 1

        # Take the action (HOLD, BUY, SELL) and update the portfolio
        self._take_action(action)

        # Calculate the new portfolio value
        self.current_portfolio_value = self._get_portfolio_value()

        # Calculate the reward
        reward = self.current_portfolio_value - self.previous_portfolio_value

        # Check if the episode is done
        done = self.current_step >= len(self.data) - 1

        # Get the next state
        state = self.data.iloc[self.current_step].values

        return state, reward, done, {}

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.portfolio = {}
        self.previous_portfolio_value = self.initial_balance
        self.current_portfolio_value = self.initial_balance

        return self.data.iloc[self.current_step].values

    def render(self, mode='human'):
        # You can implement a visualization of the trading process if desired
        pass

    def close(self):
        pass

    def _take_action(self, action):
        current_close_value = self.close_values[self.current_step]

        # Buy
        if action == 1:
            # Calculate the number of shares to buy with the current balance
            num_shares_to_buy = self.balance // current_close_value
            if num_shares_to_buy > 0:
                self.balance -= num_shares_to_buy * current_close_value * (1 + self.transaction_cost)
                if 'asset' in self.portfolio:
                    self.portfolio['asset'] += num_shares_to_buy
                else:
                    self.portfolio['asset'] = num_shares_to_buy

        # Sell
        elif action == 2:
            if 'asset' in self.portfolio and self.portfolio['asset'] > 0:
                # Sell all shares
                num_shares_to_sell = self.portfolio['asset']
                self.balance += num_shares_to_sell * current_close_value * (1 - self.transaction_cost)
                self.portfolio['asset'] = 0

        # Hold (action == 0) doesn't require any logic

    def _get_portfolio_value(self):
        asset_value = 0

        if 'asset' in self.portfolio:
            current_close_value = self.close_values[self.current_step]
            asset_value = self.portfolio['asset'] * current_close_value

        return self.balance + asset_value
