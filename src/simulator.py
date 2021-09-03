import gym
import numpy as np
import random
import math
import datetime
from gym.spaces import Box
import common


class StocksSimulator(gym.Env):
    def __init__(self, config):
        self.max_steps = config["max_steps"]
        self.n_comps = config["n_comps"]
        self.max_quotes = config["max_quotes"]

        # action space: for each company: buy confidence, buy price, sell price
        self.action_space = Box(0.0, 1.0, (3,))

        # observation space: for each company: bid/ask offers, recent quotes
        self.observation_space = Box(0.0, 1.0, (self.max_quotes,))

        self.state = []
        self.cash_init = 5000.0
        self.provision = 0.001
        self.min_transaction_value = 1000.0
        self.reset()

    def _reset(self):
        return []

    def reset(self):
        self.steps = 0
        self.done = False
        self.prices = {}
        self.timestamps = {}
        self.portfolio = {}
        self.orders = {}
        self.company = None
        self.cash = self.cash_init
        self.state = self._reset()
        self.total_reward = 0
        return self.state

    def next_state(self):
        return []

    def place_order(self, company: str, n_shares: int, buy: bool, rel_limit: float):
        """
        :param company: The company code
        :param buy: True for buy order, False for sell order
        :param rel_limit: the price limit relative to the current price
        """
        limit = round(self.prices[company][-1] * (1 + rel_limit), 2)
        self.orders[company] = {"n_shares": n_shares, "buy": buy, "limit": limit}

    def handle_orders(self):
        orders = {}
        for company, order in self.orders.items():
            price = self.prices[company][-1]
            timestamp = self.timestamps[company][-1]
            if "timestamp" not in order:
                order["timestamp"] = timestamp
            elif order["timestamp"].split()[0] < timestamp.split()[0]:
                break
            complete = False
            if order["buy"] and order["limit"] > price:
                self.portfolio[company] = {
                    "n_shares": order["n_shares"],
                    "purchase_price": order["limit"],
                }
                complete = True
            elif not order["buy"] and order["limit"] < price:
                del self.portfolio[company]
                complete = True
            if complete:
                val = order["n_shares"] * order["limit"]
                self.cash -= val * (1 if order["buy"] else -1) + self.provision * val
            else:
                orders[company] = order
        self.orders = orders

    def relative_price_decode(self, x):
        return math.pow(x - 0.5, 3)

    def relative_price_encode(self, x):
        return max(0, min(1, math.pow(abs(x), 1 / 3) * math.copysign(1, x) + 0.5))

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
        if self.steps >= self.max_steps:
            self.done = True
        company = self.company
        self.state = self.next_state()
        self.handle_orders()
        confidence = action[0] - action[1]
        rel_buy_price = self.relative_price_decode(action[1])
        rel_sell_price = self.relative_price_decode(action[2])

        if company not in self.prices:
            common.log("company:", company, ", prices:", self.prices)

        if company not in self.portfolio:
            budget = self.get_free_funds() * confidence
            comp_price = self.prices[company][-1]
            buy_price = (1 + rel_buy_price) * comp_price
            n_shares = math.floor(budget / buy_price)
            value = n_shares * comp_price
        else:
            n_shares = 0
            value = 0

        if n_shares > 0 and value >= self.min_transaction_value:
            self.place_order(company, n_shares, True, rel_buy_price)

        if company in self.portfolio:
            item = self.portfolio[company]
            n_shares = item["n_shares"]
            self.place_order(company, n_shares, False, rel_sell_price)

        capital = self.get_capital()
        if self.done:
            common.log("capital:", capital, ", reward:", self.total_reward)
        reward = capital / self.cash_init - 1
        self.total_reward += reward
        return self.state, reward, self.done, {"capital": capital, "company": self.company}


class StocksHistSimulator(StocksSimulator):
    def __init__(self, config):
        self.timestamp_format = "%Y-%m-%d %H:%M:%S"
        self.data_size = 0
        self.filename = config.get("train_file", "data/train.csv")
        with open(self.filename, "r") as f:
            for _ in f:
                self.data_size += 1
        self.data_size = self.data_size // 3
        self.prices_it = None
        self.max_steps = config.get("max_steps", 50000)
        self.n_comps = config.get("n_comps", 500)
        self.max_quotes = config.get("max_quotes", 10)
        self.max_steps = math.floor(
            min(self.max_steps / self.n_comps, self.data_size - 200) * self.n_comps
        )
        config["max_steps"] = self.max_steps
        config["n_comps"] = self.n_comps
        config["max_quotes"] = self.max_quotes
        StocksSimulator.__init__(
            self,
            config
        )

    def _reset(self):
        self.start = random.randint(
            50, self.data_size - self.max_steps // self.n_comps - 1
        )
        self.lines_processed = 0
        self.file = open(self.filename, "r")
        for _ in range(self.start * self.n_comps):
            self.next_data()
            if self.lines_processed >= self.start:
                break
        self.buffered_state = None
        self.buffered_company = None
        self.next_state()
        return self.next_state()

    def next_data(self):
        if self.prices_it is not None:
            try:
                company, prices = next(self.prices_it)
            except StopIteration:
                self.prices_it = None
        if self.prices_it is None:
            timestamp = self.file.readline().strip()
            if timestamp == "":
                self.done = True
                return None, None
            timestamp = datetime.datetime.strptime(timestamp, self.timestamp_format)
            timestamp = timestamp.strftime(self.timestamp_format)
            companies = self.file.readline().strip().split(",")
            prices = self.file.readline().strip().split(",")
            for company, price in zip(companies, prices):
                price = float(price)
                if company not in self.prices:
                    self.prices[company] = [price]
                    self.timestamps[company] = [timestamp]
                else:
                    self.prices[company].append(price)
                    self.timestamps[company].append(timestamp)
            comps = list(self.prices)[: self.n_comps]
            prices2 = {x: self.prices[x] for x in comps}
            self.prices_it = iter(prices2.items())
            company, prices = next(self.prices_it)
            self.lines_processed += 1
        return company, prices

    def next_state(self):
        company, prices = self.next_data()

        if self.done:
            self.file.close()

        if company is None:
            self.company = self.buffered_company
            return self.buffered_state

        state = []
        start = 0
        for price_i in range(self.max_quotes + 1):
            if start >= len(prices):
                break
            end = start + pow(2, price_i)
            state.append(np.mean(prices[-end : len(prices) - start]))
            start = end
        for price_i, price in enumerate(state[:-1]):
            state[price_i] = self.relative_price_encode(price / state[price_i + 1] - 1)
        state = state[:-1]
        state.reverse()
        state = [self.relative_price_encode(0)] * (self.max_quotes - len(state)) + state

        return_state = self.buffered_state
        self.company = self.buffered_company
        self.buffered_state = state
        self.buffered_company = company

        return return_state
