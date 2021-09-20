import gym
import numpy as np
import random
import math
import datetime
from gym.spaces import Box
import common
import csv

class StocksSimulator(gym.Env):
    HIST_EVENT = 0
    RT_EVENT = 1
    TRAINING = 0
    VALIDATION = 1
    PREDICTION = 2
    DT_FORMAT = "%Y-%m-%d %H:%M:%S"

    def __init__(self, config):
        self.max_steps = config["max_steps"]
        self.max_quotes = config["max_quotes"]

        # action space: for each company: buy confidence, buy price, sell price
        self.action_space = Box(0.0, 1.0, (3,))

        # observation space: for each company: bid/ask offers, recent quotes
        self.observation_space = Box(0.0, 1.0, (self.max_quotes,))

        self.state = []
        self.cash_init = 5000.0
        self.provision = 0.001
        self.min_transaction_value = 1000.0
        self.last_event_type = None
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
        if self.steps > self.max_steps:
            self.done = True
        company = self.company
        self.state, self.last_event_type = self.next_state()
        self.handle_orders()
        confidence = action[0] - action[1]
        rel_buy_price = self.relative_price_decode(action[1])
        rel_sell_price = self.relative_price_decode(action[2])

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
        return (
            self.state,
            reward,
            self.done,
            {"capital": capital, "company": self.company},
        )


class StocksRTSimulator(StocksSimulator):
    def __init__(self, config):
        self.train_max_steps = config.get("train_max_steps", 10000)
        self.validate_max_steps = config.get("validate_max_steps", 100000)
        self.max_quotes = config.get("max_quotes", 8)
        config["max_steps"] = self.max_steps
        config["max_quotes"] = self.max_quotes
        self.stage = self.TRAINING
        StocksSimulator.__init__(self, config)
        self.bar_header = ["c", "o", "h", "l", "v"]

    def _reset(self):
        if self.stage == self.TRAINING:
            self.max_steps = self.train_max_steps
        elif self.stage == self.VALIDATION:
            self.max_steps = self.validate_max_steps
        else:
            self.max_steps = 1
        if self.stage == self.PREDICTION:
            dt1 = datetime.datetime.now() - datetime.timedelta(years=1)
            dt2 = datetime.datetime.now()
        else:
            dt1 = datetime.datetime.strptime("20170901", "YYYYMMDD")
            dt2 = datetime.datetime.now()
            diff = (dt2 - dt1).days
            l = self.max_steps // 300 + 360
            shift = random.randrange(diff - l)
            dt1 += datetime.timedelta(days=shift)
            dt2 = dt1 + datetime.timedelta(days=360)
        self.dt = dt1
        self.stop_before = dt2
        self.month_loaded = None
        self.comp_iter = None
        while self.next_state() is not None:
            pass
        self.stop_before = None
        return self.next_state()

    def next_data(self):
        if self.stop_before is not None:
            self.last_event_type = self.HIST_EVENT
        else:
            if self.last_event_type == self.HIST_EVENT:
                self.last_event_type = self.RT_EVENT
            else:
                conids, prices = self.next_rt_data()
                if conids is None:
                    self.last_event_type = self.HIST_EVENT
                else:
                    self.last_event_type = self.RT_EVENT
                    return conids, prices
        if self.last_event_type == self.HIST_EVENT:
            return self.next_hist_data()
        else:
            return self.next_rt_data()

    def next_rt_data(self):
        for conid in self.watchlist:
            bar = None
            for row in self.data:
                if row["conid"] != conid:
                    continue
                dt = common.row_to_datetime(row)
                if dt > self.dt:
                    bar = {x: y for x, y in zip(self.bar_header, row)}
                    break
            

    def next_hist_data(self):
        if self.comp_iter is not None:
            try:
                return next(self.comp_iter)
            except StopIteration:
                pass
        for _ in range(10):
            month_to_load = self.dt.strftime("%Y%m")
            if self.month_loaded is None or month_to_load > self.month_loaded:
                year = self.dt.strftime("%Y")
                month = self.dt.strftime("%m")
                files = common.find_hist_quotes(year, month)
                self.data = []
                for filename in files:
                    with open(filename, "r") as f:
                        reader = csv.DictReader(f)
                        self.data.extend(list(reader))
                self.month_loaded = month_to_load
            closest_dt = None
            for row in self.data:
                dt = common.row_to_datetime(row)
                if dt >= self.dt:
                    closest_dt = dt
                    break
            if closest_dt is None:
                next_day = self.dt + datetime.timedelta(days=1)
                if next_day == self.stop_before:
                    return None, None
                self.dt = next_day
                continue
            conids = []
            for row in self.data:
                dt = common.row_to_datetime(row)
                if dt == closest_dt:
                    conid = row["conid"]
                    bar = []
                    for key in self.bar_header:
                        bar[key].append(row[key])
                    self.prices.get(conid, []).append(bar)
                    self.timestamps.get(conid, []).append(dt.strftime(self.DT_FORMAT))
                    conids.append(conid)
            self.comp_iter = iter({x: self.prices[x] for x in conids})
            break
        if conids:
            return next(self.comp_iter)
        return None, None

    def next_state(self):
        conid, prices = self.next_data()

        if conid is None:
            return None

        state = []
        start = 0
        for price_i in range(self.max_quotes + 1):
            if start >= len(prices):
                break
            end = start + pow(2, price_i)
            fragment = prices[-end : len(prices) - start]
            mean = np.array(fragment).mean(axis=0)
            state.append(mean)
            start = end
        for price_i, price in enumerate(state[:-1]):
            for var, var_i in enumerate(price):
                state[price_i][var_i] = self.relative_price_encode(var / state[price_i + 1][var_i] - 1)
        state = state[:-1]
        state.reverse()
        state = ([self.relative_price_encode(0)] * len(state[0])) * (self.max_quotes - len(state)) + state

        return state
