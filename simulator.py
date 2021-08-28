import gym
from ray.rllib.agents import sac, ppo
from custom_agent import CustomAgent
import numpy as np
import random
import math


class StocksSimulator(gym.Env):
    def __init__(self, config):
        self.max_steps = config.get("max_steps", 100)
        self.n_comps = config.get("n_comps", 500)
        self.max_quotes = config.get("max_quotes", 25)

        # action space: for each company: buy confidence, buy price, sell price
        self.action_space = [gym.spaces.Box(0.0, 1.0, (self.n_comps, 3))]

        # observation space: for each company: bid/ask offers, recent quotes
        self.observation_space = gym.spaces.Box(
            -10.0, 10.0, (self.n_comps, self.max_quotes)
        )

        self.state = None
        self.steps = None

        self.prices = {}
        self.timestamps = {}
        self.portfolio = {}
        self.orders = {}
        self.cash_init = 5000.0
        self.cash = self.cash_init
        self.provision = .001
        self.min_transaction_value = 1000.0

    def place_order(self, company: str, n_shares: int, buy: bool, limit: float):
        """
        :param company: The company code
        :param buy: True for buy order, False for sell order
        :param limit: the price limit relative to the current price
        """
        limit = round(self.prices[company][-1] * (1 + limit), 2)
        self.orders[company] = {"n_shares": n_shares, "buy": buy, "limit": limit}

    def handle_orders(self):
        for company, order in self.orders.items():
            price = self.prices[company][-1]
            timestamp = self.timestamps[company][-1]
            if "timestamp" not in order:
                order["timestamp"] = timestamp
            elif order["timestamp"].split()[0] < timestamp.split()[0]:
                del self.orders[company]
                break
            complete = False
            if order["buy"] and order["limit"] > price:
                complete = True
            elif not order["buy"] and order["limit"] < price:
                complete = True
            if complete:
                self.portfolio[company] = {
                    "n_shares": order["n_shares"],
                    "purchase_price": order["limit"],
                }
                val = order["n_shares"] * order["limit"]
                self.cash -= val * (1 if order["buy"] else -1) + self.provision * val
                del self.orders[company]

    def _reset(self):
        return []

    def reset(self):
        self.state = self._reset()
        self.steps = 0
        return self.state

    def next_state(self):
        return []

    def relative_price_decode(self, x):
        return math.pow(x-.5, 3)

    def relative_price_encode(self, x):
        return math.pow(abs(x), 1/3) * math.copysign(1, x) + .5

    def get_capital(self):
        capital = self.cash
        for company, item in self.portfolio.items():
            price = self.prices[company][-1]
            capital += item["n_shares"] * price
        return capital

    def get_blocked_cash(self):
        blocked = 0
        for company, order in self.orders.items():
            if order["buy"]:
                blocked += order["limit"] * order["n_shares"]
        return blocked

    def get_free_funds(self):
        return self.cash - self.get_blocked_cash()

    def step(self, action):
        self.steps += 1
        self.state = self.next_state()
        self.handle_orders()
        best_comp_i = None
        for _ in range(10):
            best_comp_i = np.argmax([x[0] for x in action])
            best_comp = list(self.prices)[best_comp_i]
            if best_comp not in self.portfolio:
                break
            else:
                best_comp_i = None
        if best_comp_i is not None:
            budget = self.get_free_funds() * action[best_comp_i][0]
            best_comp_price = self.prices[best_comp][-1]
            n_shares = math.floor(budget / best_comp_price)
            value = n_shares * best_comp_price
        else:
            n_shares = 0
            value = 0
        if n_shares > 0 and value >= self.min_transaction_value:
            buy_price = self.relative_price_decode(action[best_comp_i][1])
            self.place_order(best_comp, n_shares, True, buy_price)
        for company, item in self.portfolio.items():
            n_shares = item["n_shares"]
            sell_price = self.relative_price_decode(action[best_comp_i][2])
            self.place_order(company, n_shares, False, sell_price)
        done = self.steps >= self.max_steps
        reward = self.get_capital() / self.cash_init
        return self.state, reward, done, {}


class StocksHistSimulator(StocksSimulator):
    def __init__(self, config):
        StocksSimulator.__init__(self, config)
        self.data_size = 0
        self.filename = "data/all_hist.csv"
        with open(self.filename, "r") as f:
            for _ in f:
                self.data_size += 1
        self.max_steps = min(self.max_steps, self.data_size - 200)

    def _reset(self):
        start = random.randint(50, self.data_size - self.max_steps)
        self.file = open(self.filename, "r")
        for _ in range(start):
            self.next_data()
        return self.next_state()

    def next_data(self):
        timestamp = self.file.readline().strip()
        companies = self.file.readline().strip().split(",")
        prices = self.file.readline().strip().split(",")
        for company, price in zip(companies, prices):
            if company not in self.prices:
                self.prices[company] = [price]
                self.timestamps[company] = [timestamp]
            else:
                self.prices[company].append(price)
                self.timestamps[company].append(timestamp)

    def next_state(self):
        state = []
        self.next_data()

        for company in self.prices:
            prices = self.prices[company]
            compressed = []
            start = 0
            for price_i in range(self.max_quotes + 1):
                if start >= len(prices):
                    break
                end = start + pow(2, price_i // 5)
                compressed.append(np.mean(prices[-end : len(prices) - start]))
                start = end
            for price_i, price in enumerate(compressed[:-1]):
                compressed[price_i] = price / compressed[price_i + 1] - 1
            compressed = compressed[:-1]
            compressed.reverse()
            compressed = [0] * (self.max_quotes - len(compressed)) + compressed
            state.append(compressed)

        state = state + [[0] * self.max_quotes] * (self.n_comps - len(state))

        if self.steps >= self.max_steps:
            self.file.close()

        return state
