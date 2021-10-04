import datetime
import gym
import random
import numpy as np
from gym.spaces.discrete import Discrete
import tensorflow.keras as keras
import pickle
import common
from tensorflow import saved_model
import glob
import shutil


class CustomAgent:
    def __init__(
        self, env: gym.Env, config: dict = {}, env_config: dict = {}, worker_id=None
    ):
        self.env = env(env_config)
        self.train_max_steps = self.env.train_max_steps
        self.validate_max_steps = self.env.validate_max_steps
        self.max_quotes = self.env.max_quotes
        self.hist_model = self.create_hist_model()
        self.rt_model = self.create_rt_model()
        self.avg_reward = 0
        self.avg_total = None
        self.std_total = None
        self.max_total = None
        self.avg_profit = None
        self.std_profit = None
        self.explore = 1
        self.fitted = False
        self.best_score = None
        self.model_dir = "data"
        self.model_changed = False
        self.worker_id = (
            worker_id
            if worker_id is not None
            else datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        )
        self.confidences = {}
        self.niter = 0

    def create_hist_model(self):
        inputs = keras.layers.Input(shape=(self.max_quotes, 5))
        l = inputs
        l = keras.layers.LSTM(self.max_quotes)(l)
        l = keras.layers.Dense(100, activation="relu")(l)
        l = keras.layers.Dense(100, activation="relu")(l)
        l = keras.layers.Dense(50, activation="relu")(l)
        l = keras.layers.Dense(20, activation="relu")(l)
        l = keras.layers.Dense(10, activation="relu")(l)
        outputs = keras.layers.Dense(1, activation="sigmoid")(l)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Nadam(learning_rate=0.001),
            loss="mean_squared_logarithmic_error",
        )
        return model

    def create_rt_model(self):
        inputs = keras.layers.Input(shape=(self.max_quotes, 6))
        l = inputs
        l = keras.layers.LSTM(self.max_quotes)(l)
        l = keras.layers.Dense(100, activation="relu")(l)
        l = keras.layers.Dense(100, activation="relu")(l)
        l = keras.layers.Dense(50, activation="relu")(l)
        l = keras.layers.Dense(20, activation="relu")(l)
        l = keras.layers.Dense(10, activation="relu")(l)
        outputs = keras.layers.Dense(3, activation="sigmoid")(l)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Nadam(learning_rate=0.001),
            loss="mean_squared_logarithmic_error",
        )
        return model

    def predict_action(self, x):
        if self.env.last_event_type == self.env.HIST_EVENT:
            confidence = self.hist_model.predict_on_batch(
                np.array([x]).astype(np.float32)
            )[0][0]
            action = [confidence, 0.5, 0.5]
            self.confidences[self.env.company] = confidence
            # common.log("hist pred:", confidence)
        else:
            confidence, buy_price, sell_price = self.rt_model.predict_on_batch(
                np.array([x]).astype(np.float32)
            )[0]
            confidence = (
                confidence + self.confidences.get(self.env.company, confidence)
            ) / 2
            action = [confidence, buy_price, sell_price]
            # common.log("rt pred:", confidence, buy_price, sell_price)
        return action

    def fit(self, model, x, y, weights):
        common.log("train set size:", len(x))
        if len(x) == 0:
            return
        model.fit(
            np.array(x).astype(np.float32),
            np.array(y).astype(np.float32),
            epochs=10,
            verbose=0,
            sample_weight=np.array(weights),
        )

    def transform_x(self, state):
        return state

    def transform_y(self, action):
        if self.env.last_event_type == self.env.HIST_EVENT:
            return action[0]
        else:
            return action

    def run_episode(self, train=True, live=False):
        stage = (
            self.env.TRAINING
            if train
            else (self.env.VALIDATION if not live else self.env.PREDICTION)
        )
        self.env.stage = stage
        state = self.env.reset()
        hist_set = {
            "all_x": [],
            "all_y": [],
            "train_x": [],
            "train_y": [],
            "company": [],
            "dt": [],
            "weights": [],
        }
        rt_set = {
            "all_x": [],
            "all_y": [],
            "train_x": [],
            "train_y": [],
            "company": [],
            "dt": [],
            "weights": [],
        }
        total = 0
        max_steps = self.train_max_steps if train else self.validate_max_steps
        seed = random.gauss(0, self.explore * 0.1)
        for _ in range(max_steps + 1):
            self.niter += 1
            x = self.transform_x(state)
            if self.env.last_event_type == self.env.HIST_EVENT:
                trainset = hist_set
            else:
                trainset = rt_set
            if self.fitted:
                action = self.predict_action(x)
            else:
                action = self.env.action_space.sample()
            if train and random.random() < self.explore:
                for val_i, val in enumerate(action):
                    action[val_i] = min(
                        1, max(0, val + seed + random.gauss(0, self.explore * 0.1))
                    )
            if train:
                y = self.transform_y(action)
                trainset["all_x"].append(x)
                trainset["all_y"].append(y)
                trainset["company"].append(self.env.company)
                trainset["dt"].append(self.env.dt)
            state, reward, done, info = self.env.step(action)
            total += reward
            self.avg_reward = self.avg_reward * 0.99 + reward * 0.01
            if done:
                break
        self.std_total = (
            self.std_total * 0.9 + abs(total - self.avg_total) * 0.1
            if self.avg_total is not None
            else 0
        )
        self.avg_total = (
            self.avg_total * 0.9 + total * 0.1 if self.avg_total is not None else total
        )
        if train:
            common.log("Fit")
            nit = max(
                1,
                round((total - self.avg_total + 1) / (self.std_total + 1)),
            )
            nit = 1
            # common.log("nit:", nit)
            good_bad_trans = []
            n_good = 0
            n_bad = 0
            n_neutral = 0
            for trans in self.env.transactions:
                if not trans["buy"] and "profit_percent" in trans:
                    self.std_profit = (
                        self.std_profit * 0.999
                        + abs(trans["profit_percent"] - self.avg_profit) * 0.001
                        if self.std_profit is not None
                        else 0
                    )
                    self.avg_profit = (
                        self.avg_profit * 0.999 + trans["profit_percent"] * 0.001
                        if self.avg_profit is not None
                        else trans["profit_percent"]
                    )
                    if trans["profit_percent"] > max(
                        0, self.avg_profit + 0.5 * self.std_profit
                    ):
                        good_bad_trans.append((trans, True))
                        n_good += 1
                    elif trans["profit_percent"] < 0:
                        good_bad_trans.append((trans, False))
                        n_bad += 1
                    else:
                        n_neutral += 1
            common.log("Good:", n_good, "Bad:", n_bad, "Neutral:", n_neutral)
            for trans, good in good_bad_trans:
                buy_dt = trans["buy_transaction"]["buy_dt"]
                sell_dt = trans["sell_dt"]
                hist_train_x = None
                hist_train_y = None
                buy_train_x = None
                buy_train_y = None
                sell_train_x = None
                sell_train_y = None
                for trainset in [hist_set, rt_set]:
                    for dt, company, x, y in zip(
                        trainset["dt"],
                        trainset["company"],
                        trainset["all_x"],
                        trainset["all_y"],
                    ):
                        if (
                            trainset is hist_set
                            and company == trans["company"]
                            and dt < buy_dt
                        ):
                            hist_train_x = x
                            hist_train_y = int(good)
                        elif (
                            trainset is rt_set
                            and company == trans["company"]
                            and dt < buy_dt
                        ):
                            if good:
                                y = [1, y[1], y[2]]
                            else:
                                y = [0, 1 - y[1], 1 - y[2]]
                            buy_train_x = x
                            buy_train_y = y
                        elif (
                            trainset is rt_set
                            and company == trans["company"]
                            and dt < sell_dt
                        ):
                            if good:
                                y = [1, y[1], y[2]]
                            else:
                                y = [0, 1 - y[1], 1 - y[2]]
                            sell_train_x = x
                            sell_train_y = y
                        elif (trainset is hist_set and dt >= buy_dt) or (
                            trainset is rt_set and dt >= sell_dt
                        ):
                            break
                w = int(good) * 0.8 + 0.2
                if hist_train_x is not None:
                    hist_set["train_x"].append(hist_train_x)
                    hist_set["train_y"].append(hist_train_y)
                    hist_set["weights"].append(w)
                if buy_train_x is not None:
                    rt_set["train_x"].append(buy_train_x)
                    rt_set["train_y"].append(buy_train_y)
                    rt_set["weights"].append(w)
                if sell_train_x is not None:
                    rt_set["train_x"].append(sell_train_x)
                    rt_set["train_y"].append(sell_train_y)
                    rt_set["weights"].append(w)
            for _ in range(nit):
                self.fit(
                    self.hist_model,
                    hist_set["train_x"],
                    hist_set["train_y"],
                    hist_set["weights"],
                )
                self.fit(
                    self.rt_model,
                    rt_set["train_x"],
                    rt_set["train_y"],
                    rt_set["weights"],
                )
            self.fitted = True
            self.explore = max(0.3, self.explore * 0.99)
        if self.max_total is None or total >= self.max_total:
            self.max_total = total
        return total

    def train(self):
        self.niter = 0
        self.max_total = None
        for _ in range(1000):
            self.run_episode()
            if self.niter > 30000:
                break
        common.log(
            "avg total:",
            self.avg_total,
            "max total:",
            self.max_total,
            ", avg profit:",
            self.avg_profit,
            ", std profit:",
            self.std_profit,
            "explore:",
            self.explore,
        )

    def evaluate(self, quick=False, find_best=False):
        if quick:
            total = self.max_total
        else:
            self.env.log_transactions = True
            total = self.run_episode(train=False)
        common.log("Best score:", self.best_score)
        common.log("Current score:", total)
        if self.best_score is None or total >= self.best_score or not quick:
            self.best_score = total
        self.save_checkpoint(self.model_dir)
        if find_best:
            global_best = None
            for agent_file in glob.iglob(self.model_dir + "/agent-????*.dat"):
                agent_data = pickle.load(open(agent_file, "rb"))
                common.log(agent_file, agent_data["best_score"])
                if global_best is None or agent_data["best_score"] > global_best:
                    best_agent = agent_file
                    global_best = agent_data["best_score"]
            if "agent-best" not in best_agent:
                common.log(best_agent, "->", self.model_dir + "/agent-best.dat")
                shutil.copyfile(best_agent, self.model_dir + "/agent-best.dat")
                hist_model_file = best_agent.replace("agent", "hist_model").replace(
                    ".dat", ".h5"
                )
                common.log(
                    hist_model_file, "->", self.model_dir + "/hist_model-best.dat"
                )
                shutil.copyfile(
                    hist_model_file, self.model_dir + "/hist_model-best.dat"
                )
                rt_model_file = best_agent.replace("agent", "rt_model").replace(
                    ".dat", ".h5"
                )
                common.log(rt_model_file, "->", self.model_dir + "/rt_model-best.dat")
                shutil.copyfile(rt_model_file, self.model_dir + "/rt_model-best.dat")
        return total

    def __getstate__(self) -> dict:
        keras.models.save_model(
            self.hist_model,
            f"{self.model_dir}/hist_model-{self.worker_id}.h5",
            save_format="h5",
        )
        keras.models.save_model(
            self.rt_model,
            f"{self.model_dir}/rt_model-{self.worker_id}.h5",
            save_format="h5",
        )
        return {
            "best_score": self.best_score,
            "explore": self.explore,
            "fitted": self.fitted,
            "avg_total": self.avg_total,
            "std_total": self.std_total,
            "avg_profit": self.avg_profit,
            "std_profit": self.std_profit,
        }

    def __setstate__(self, state: dict, worker_id=None):
        hist_model_file = f"{self.model_dir}/hist_model.h5"
        rt_model_file = f"{self.model_dir}/rt_model.h5"
        if worker_id is not None:
            hist_model_file = hist_model_file.replace(".h5", f"-{worker_id}.h5")
            rt_model_file = rt_model_file.replace(".h5", f"-{worker_id}.h5")
        self.hist_model = keras.models.load_model(hist_model_file, compile=True)
        self.rt_model = keras.models.load_model(rt_model_file, compile=True)
        self.best_score = state["best_score"]
        self.explore = state["explore"]
        self.fitted = state["fitted"]
        self.avg_total = state.get("avg_total")
        self.std_total = state.get("std_total")
        self.avg_profit = state.get("avg_profit")
        self.std_profit = state.get("std_profit")

    def save_checkpoint(self, checkpoint_dir: str = None) -> str:
        if checkpoint_dir is None:
            checkpoint_dir = self.model_dir
        checkpoint_path = f"{checkpoint_dir}/agent-{self.worker_id}.dat"
        pickle.dump(self.__getstate__(), open(checkpoint_path, "wb"))
        self.model_changed = True
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str):
        worker_id = None
        if "-" in checkpoint_path:
            worker_id = checkpoint_path.split("-")[-1].split(".")[0]
        extra_data = pickle.load(open(checkpoint_path, "rb"))
        self.__setstate__(extra_data, worker_id)
