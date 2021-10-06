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
        self.max_quotes = config["max_quotes"]
        self.watchlist_size = config.get("watchlist_size", 5)
        self.confidence_th = config.get("confidence_th", 0.2)

        # action space: for each company: buy confidence, buy price, sell price
        self.action_space = Box(0.0, 1.0, (3,))

        # observation space: for each company: bid/ask offers, recent quotes
        self.observation_space = Box(0.0, 1.0, (self.max_quotes,))

        self.state = []
        self.cash_init = 5000.0
        self.provision = 0.001
        self.min_transaction_value = 1000.0
        self.last_event_type = None
        self.log_transactions = False
        # self.reset()

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
        self.capital = self.cash
        self.watchlist = []
        self.state = self._reset()
        self.total_reward = 0
        self.n_bought = 0
        self.n_sold = 0
        self.transactions = []
        self.prev_dt = None
        self.avg_confidence = None
        self.std_confidence = None
        self.first_day = True
        return self.state

    def next_state(self):
        return []

    def place_order(self, company: str, n_shares: int, buy: bool, rel_limit: float):
        """
        :param company: The company code
        :param n_shares: The number of shares to buy/sell
        :param buy: True for buy order, False for sell order
        :param rel_limit: the price limit relative to the current price
        """
        common.log_debug("place order:", company, n_shares, buy, rel_limit)
        limit = round(self.get_current_price(company) * (1 + rel_limit), 2)
        self.orders[company] = {
            "n_shares": n_shares,
            "buy": buy,
            "limit": limit,
            "order_dt": self.dt,
        }

    def handle_orders(self):
        if not 14 <= self.dt.hour <= 23:
            return
        orders = {}
        for company, order in self.orders.items():
            if order["order_dt"] >= self.dt:
                continue
            if "session_dt" not in order:
                order["session_dt"] = self.dt.date()
            elif order["session_dt"] < self.dt.date():
                break
            complete = False
            price = self.get_current_price(company)
            if not 0.9 < order["limit"] / price < 1.1:
                continue
            if order["buy"] and order["limit"] > price:
                self.portfolio[company] = {
                    "n_shares": order["n_shares"],
                    "purchase_price": order["limit"],
                }
                order["company"] = company
                order["buy_dt"] = self.dt
                self.transactions.append(order)
                complete = True
                self.n_bought += 1
                if self.log_transactions:
                    common.log(
                        "buy:",
                        "limit",
                        order["limit"],
                        "n_shares",
                        order["n_shares"],
                        "buy_dt",
                        order["buy_dt"],
                        "company",
                        order["company"],
                    )
            elif not order["buy"] and order["limit"] < price:
                del self.portfolio[company]
                for trans in reversed(self.transactions):
                    if trans["company"] == company:
                        order["buy_transaction"] = trans
                        order["profit_percent"] = order["limit"] / trans["limit"] - 1
                        break
                order["company"] = company
                order["sell_dt"] = self.dt
                self.transactions.append(order)
                complete = True
                self.n_sold += 1
                if self.log_transactions:
                    common.log(
                        "sell:",
                        "limit",
                        order["limit"],
                        "n_shares",
                        order["n_shares"],
                        "sell_dt",
                        order["sell_dt"],
                        "company",
                        order["company"],
                    )
            if complete:
                val = order["n_shares"] * order["limit"]
                self.cash -= val * (1 if order["buy"] else -1) + self.provision * val
                if self.log_transactions:
                    common.log("capital", self.get_capital())
            else:
                orders[company] = order
        self.orders = orders

    def relative_price_decode(self, x):
        return math.pow(x - 0.5, 3)

    def relative_price_encode(self, x):
        return max(0, min(1, math.pow(abs(x), 1 / 3) * math.copysign(1, x) + 0.5))

    def get_current_price(self, company):
        if (
            self.last_event_type == self.HIST_EVENT
            or company not in self.rt_prices
            or not self.rt_prices[company]
        ):
            return self.prices[company][-1][0]
        else:
            return self.rt_prices[company][-1][0]

    def get_capital(self):
        capital = self.cash
        for company, item in self.portfolio.items():
            price = self.get_current_price(company)
            item["last_price"] = price
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
        self.handle_orders()
        confidence = action[0]
        rel_buy_price = self.relative_price_decode(action[1] - 0.2)
        rel_sell_price = self.relative_price_decode(action[2] + 0.2)
        self.std_confidence = (
            self.std_confidence * 0.999 + abs(confidence - self.avg_confidence) * 0.001
            if self.avg_confidence is not None and self.std_confidence is not None
            else 0
        )
        self.avg_confidence = (
            self.avg_confidence * 0.999 + confidence * 0.001
            if self.avg_confidence is not None
            else confidence
        )

        if self.last_event_type == self.HIST_EVENT:
            if self.prev_dt is None or self.dt.day != self.prev_dt.day:
                self.watchlist = []
                self.n_processed = 0
                if self.prev_dt is not None:
                    self.first_day = False
                # common.log("Clear watchlist")
            self.prev_dt = self.dt
            self.n_processed += 1

            for company in self.portfolio:
                if company not in self.watchlist:
                    self.watchlist.append(company)

            if (
                len(self.watchlist) < self.watchlist_size
                and confidence > self.confidence_th
                and confidence > self.avg_confidence + 3 * self.std_confidence
                and not self.first_day
                and self.company not in self.watchlist
            ):
                self.watchlist.append(self.company)
                common.log("Hist comp processed:", self.n_processed)

            # common.log(self.dt, "watchlist size:", len(self.watchlist))

        if self.last_event_type == self.RT_EVENT:
            if self.company in self.portfolio:
                item = self.portfolio[self.company]
                n_shares = item["n_shares"]
                self.place_order(self.company, n_shares, False, rel_sell_price)
            else:
                budget = self.get_free_funds() * confidence
                comp_price = self.get_current_price(self.company)
                buy_price = (1 + rel_buy_price) * comp_price
                n_shares = math.floor(budget / buy_price)
                value = n_shares * comp_price
                if n_shares > 0 and value >= self.min_transaction_value:
                    self.place_order(self.company, n_shares, True, rel_buy_price)

        self.state = self.next_state()
        if self.state is None:
            self.done = True
        self.capital = self.get_capital()
        if self.done:
            common.log("Finish simulation on:", self.dt.strftime(self.DT_FORMAT))
            common.log("capital:", self.capital, ", reward:", self.total_reward)
        reward = (self.capital / self.cash_init - 1) * self.steps / 10000
        if self.capital > 20000:
            common.log("dt:", self.dt)
            common.log("capital:", self.capital)
            common.log("portfolio:", self.portfolio)
            common.log("watchlist:", self.watchlist)
            common.log("cash:", self.cash)
            for company in self.portfolio:
                common.log("company:", company)
                if company in self.rt_prices:
                    common.log([x[0] for x in self.prices[company][-3:]])
                    common.log([x[0] for x in self.rt_prices[company][:5]])
                    common.log([x[0] for x in self.rt_prices[company][-5:]])
                    if company in self.rt_scale:
                        common.log("rt_scale:", self.rt_scale[company])
                    if company in self.rt_shift:
                        common.log("rt_shift:", self.rt_shift[company])
                    if company in self.rt_conid2:
                        common.log("rt_conid2:", self.rt_conid2[company])
                    for conid, row, hour, complete in self.rt_data:
                        if conid == company:
                            common.log(row)
                    if company in self.bars:
                        common.log("bars:", self.bars[company])
        self.total_reward += reward
        return (
            self.state,
            reward,
            self.done,
            {},
        )


class StocksRTSimulator(StocksSimulator):
    def __init__(self, config):
        self.train_max_steps = config.get("train_max_steps", 60000)
        self.validate_max_steps = config.get("validate_max_steps", 100000)
        self.max_quotes = config.get("max_quotes", 8)
        config["train_max_steps"] = self.train_max_steps
        config["validate_max_steps"] = self.validate_max_steps
        config["max_quotes"] = self.max_quotes
        self.stage = self.TRAINING
        self.bar_header = ["c", "o", "h", "l", "v"]
        self.rt_header = {
            "price": "31",
            "volume": "7762",
            "bid": "84",
            "ask_size": "85",
            "ask": "86",
            "bid_size": "88",
        }
        StocksSimulator.__init__(self, config)

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
        elif self.stage == self.TRAINING:
            dt1 = datetime.datetime.strptime("20170901", "%Y%m%d")
            dt2 = datetime.datetime.now()
            diff = (dt2 - dt1).days
            l = self.max_steps // 300 + 360
            shift = random.randrange(diff - l)
            dt1 += datetime.timedelta(days=shift)
            dt2 = dt1 + datetime.timedelta(days=360)
        else:
            dt2 = datetime.datetime.now() - datetime.timedelta(
                days=self.max_steps // 300
            )
            dt1 = dt2 - datetime.timedelta(days=360)
        self.dt = dt1
        common.log("Start preparation on:", self.dt.strftime(self.DT_FORMAT))
        self.stop_before = dt2
        self.month_loaded = None
        self.comp_iter = None
        self.rt_data = None
        self.bars = None
        self.rt_prices = {}
        while self.next_state() is not None:
            pass
        self.stop_before = None
        common.log("Start simulation on:", self.dt.strftime(self.DT_FORMAT))
        return self.next_state()

    def next_data(self):
        if self.stop_before is not None or not self.watchlist:
            self.last_event_type = self.HIST_EVENT
        else:
            if self.last_event_type == self.HIST_EVENT and self.eod:
                self.last_event_type = self.RT_EVENT
            if self.last_event_type == self.RT_EVENT:
                conid, prices = self.next_rt_data()
                if conid is None:
                    self.last_event_type = self.HIST_EVENT
                else:
                    return conid, prices
        return self.next_hist_data()

    def next_rt_data(self):
        if self.bars is None:
            self.bars = {}
            for conid in self.watchlist:
                for row in self.data:
                    if row["conid"] != conid:
                        continue
                    dt = common.row_to_datetime(row)
                    if dt > self.dt:
                        bar = {}
                        for key in self.bar_header:
                            bar[key] = common.price_to_float(row[key])
                        self.bars[conid] = bar
                        break
        if self.rt_data is None:
            self.rt_data = []
            self.rt_prices = {}
            self.rt_scale = {}
            self.rt_shift = {}
            self.vol_scale = {}
            self.size_scale = {}
            self.rt_conid2 = {}
            for conid in self.watchlist:
                self.rt_prices[conid] = [
                    [self.prices[conid][-1][0]] + [0] * (len(self.rt_header) - 1)
                ]
                filename = common.find_random_rt_quotes()
                with open(filename, "r") as f:
                    reader = csv.DictReader(f)
                    data = list(reader)
                row = random.choice(data)
                conidex = row["conidex"]
                self.rt_conid2[conid] = conidex
                first_price = None
                for row_i, row in enumerate(data):
                    if row["conidex"] == conidex:
                        dt = common.row_to_datetime(row)
                        hour = dt.strftime("%H%M%S")
                        complete = (row_i + 1) / len(data)
                        self.rt_data.append((conid, row, hour, complete))
                        if first_price is None:
                            first_price = row[self.rt_header["price"]]
                        last_price = row[self.rt_header["price"]]
                        volume = row[self.rt_header["volume"]]
                first_price = common.price_to_float(first_price)
                last_price = common.price_to_float(last_price)
                volume = common.price_to_float(volume)
                if conid in self.bars:
                    self.rt_scale[conid] = self.bars[conid]["o"] / first_price
                    self.rt_shift[conid] = (
                        self.bars[conid]["c"] - last_price * self.rt_scale[conid]
                    )
                    self.vol_scale[conid] = self.bars[conid]["v"] / volume
                else:
                    self.rt_scale[conid] = self.prices[conid][-1][0] / first_price
                    self.rt_shift[conid] = 0
                    self.vol_scale[conid] = (
                        self.prices[conid][-1][self.bar_header.index("v")] / volume
                    )
                self.size_scale[conid] = first_price / volume
            self.rt_data.sort(key=lambda x: x[2])
            self.rt_iter = iter(self.rt_data)
        try:
            conid, row, hour, complete = next(self.rt_iter)
            rt = []
            for key, val in self.rt_header.items():
                x = common.price_to_float(row[val])
                if key in ["price", "ask", "bid"]:
                    x *= self.rt_scale[conid]
                    x += complete * self.rt_shift[conid]
                elif key == "volume":
                    x *= self.vol_scale[conid]
                else:
                    x *= self.size_scale[conid]
                rt.append(x)
            self.rt_prices[conid].append(rt)
            self.dt = self.dt.replace(
                hour=int(hour[:2]), minute=int(hour[2:4]), second=int(hour[4:])
            )
            return conid, self.rt_prices[conid]
        except StopIteration:
            self.rt_data = None
            self.bars = None
            return None, None

    def next_hist_data(self):
        self.eod = False
        if self.comp_iter is not None:
            try:
                conid, prices = next(self.comp_iter)
                self.eod = conid == self.last_hist_conid
                return conid, prices
            except StopIteration:
                self.comp_iter = None
        conids = []
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
                self.data.sort(key=lambda x: x["t"])
                self.month_loaded = month_to_load
            closest_dt = None
            for row in self.data:
                dt = common.row_to_datetime(row)
                if dt > self.dt:
                    closest_dt = dt
                    break
            if closest_dt is None:
                self.dt = self.dt + datetime.timedelta(days=1)
                self.dt.replace(hour=0)
                continue
            else:
                self.dt = closest_dt
            if self.stop_before is not None and self.dt >= self.stop_before:
                return None, None
            for row in self.data:
                dt = common.row_to_datetime(row)
                if dt == closest_dt:
                    conid = row["conid"]
                    bar = []
                    for key in self.bar_header:
                        bar.append(common.price_to_float(row[key]))
                    if conid in self.prices:
                        self.prices[conid].append(bar)
                        self.timestamps[conid].append(dt.strftime(self.DT_FORMAT))
                    else:
                        self.prices[conid] = [bar]
                        self.timestamps[conid] = [dt.strftime(self.DT_FORMAT)]
                    conids.append(conid)
                elif dt > closest_dt:
                    break
            if len(conids) >= 100:
                break
        if conids:
            random.shuffle(conids)
            self.last_hist_conid = conids[-1]
            self.comp_iter = iter({x: self.prices[x] for x in conids}.items())
        else:
            self.comp_iter = None
        if self.comp_iter is not None:
            return next(self.comp_iter)
        return None, None

    def next_state(self):
        conid, prices = self.next_data()

        if conid is None:
            return None

        self.company = conid
        state = []
        start = 0
        for price_i in range(self.max_quotes + 1):
            if start >= len(prices):
                break
            end = start + pow(2, price_i)
            fragment = prices[-end : len(prices) - start]
            mean = np.array(fragment).mean(axis=0).tolist()
            state.append(mean)
            start = end
        for price_i, price in enumerate(state[:-1]):
            for var_i, var in enumerate(price):
                if abs(state[price_i + 1][var_i]) > 0.001:
                    state[price_i][var_i] = self.relative_price_encode(
                        var / state[price_i + 1][var_i] - 1
                    )
                else:
                    state[price_i][var_i] = 0
        if len(state) > 1:
            state = state[:-1]
        else:
            state = [[self.relative_price_encode(0)] * len(state[0])]
        state.reverse()
        state = (
            np.full(
                (self.max_quotes - len(state), len(state[0])),
                self.relative_price_encode(0),
            ).tolist()
            + state
        )

        return state
