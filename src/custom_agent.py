import gym
import random
import numpy as np
from gym.spaces.discrete import Discrete
import tensorflow.keras as keras
import pickle
import common
from tensorflow import saved_model


class CustomAgent:
    def __init__(
        self, env: gym.Env, config: dict = {}, env_config: dict = {}, worker_id=0
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
        self.explore = 1
        self.fitted = False
        self.best_score = None
        self.model_dir = "data"
        self.model_changed = False
        self.worker_id = worker_id
        self.confidences = {}

    def create_hist_model(self):
        inputs = keras.layers.Input(shape=(self.max_quotes,5))
        l = inputs
        l = keras.layers.LSTM(self.max_quotes)(l)
        l = keras.layers.Dense(100, activation="relu")(l)
        l = keras.layers.Dense(10, activation="relu")(l)
        l = keras.layers.Dense(10, activation="relu")(l)
        l = keras.layers.Dense(10, activation="relu")(l)
        outputs = keras.layers.Dense(1, activation="sigmoid")(l)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Nadam(learning_rate=0.001),
            loss="mean_squared_logarithmic_error",
        )
        return model

    def create_rt_model(self):
        inputs = keras.layers.Input(shape=(self.max_quotes,6))
        l = inputs
        l = keras.layers.LSTM(self.max_quotes)(l)
        l = keras.layers.Dense(100, activation="relu")(l)
        l = keras.layers.Dense(10, activation="relu")(l)
        l = keras.layers.Dense(10, activation="relu")(l)
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
            confidence = self.hist_model.predict_on_batch(np.array([x]).astype(np.float32))[0][0]
            action = [confidence, 0.5, 0.5]
            self.confidences[self.env.company] = confidence
        else:
            confidence, buy_price, sell_price = self.rt_model.predict_on_batch(np.array([x]).astype(np.float32))[0]
            confidence = (confidence + self.confidences.get(self.env.company, confidence)) / 2
            action = [confidence, buy_price, sell_price]
        if type(self.env.action_space) == Discrete:
            if np.shape(action) == (1,):
                action = action[0]
            action = round(action)
            action = max(0, action)
            action = min(self.env.action_space.n - 1, action)
        return action

    def fit(self, model, x, y):
        model.fit(
            np.array(x).astype(np.float32),
            np.array(y).astype(np.float32),
            epochs=10,
            verbose=0,
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
            "train_x": [],
            "train_y": [],
        }
        rt_set = {
            "train_x": [],
            "train_y": [],
        }
        total = 0
        max_steps = self.train_max_steps if train else self.validate_max_steps
        for _ in range(max_steps + 1):
            self.niter += 1
            x = self.transform_x(state)
            if self.env.last_event_type == self.env.HIST_EVENT:
                trainset = hist_set
            else:
                trainset = rt_set
            if (train and random.random() < self.explore) or not self.fitted:
                action = self.env.action_space.sample()
            else:
                action = self.predict_action(x)
            if train:
                y = self.transform_y(action)
                trainset["train_x"].append(x)
                trainset["train_y"].append(y)
            state, reward, done, info = self.env.step(action)
            total += reward
            self.avg_reward = self.avg_reward * 0.99 + reward * 0.01
            if done:
                break
        self.std_total = self.std_total * 0.9 + abs(total - self.avg_total) * 0.1 if self.avg_total is not None else 0
        self.avg_total = self.avg_total * 0.9 + total * 0.1 if self.avg_total is not None else total
        common.log(train, total, self.avg_total, self.std_total)
        if train and total > self.avg_total + .8 * self.std_total:
            common.log("Fit")
            nit = max(
                1,
                round((total - self.avg_total + 1) / (self.std_total + 1) - 10),
            )
            common.log("nit:", nit)
            for _ in range(nit):
                self.fit(self.hist_model, hist_set["train_x"], hist_set["train_y"])
                self.fit(self.rt_model, rt_set["train_x"], rt_set["train_y"])
            self.fitted = True
            self.explore = max(0.3, self.explore * 0.9999)
        if self.max_total is None or total >= self.max_total:
            self.max_total = total
        return total

    def train(self):
        self.niter = 0
        self.max_total = None
        for _ in range(1000):
            self.run_episode()
            if self.niter > 100000:
                break
        common.log(
            "avg r:",
            self.avg_reward,
            ", avg total:",
            self.avg_total,
            ", std total:",
            self.std_total,
            "max:",
            self.max_total,
            "explore:",
            self.explore,
        )

    def evaluate(self, quick=False):
        if quick:
            total = self.max_total
        else:
            total = self.run_episode(train=False)
        common.log("Best score:", self.best_score)
        common.log("Current score:", total)
        self.save_checkpoint(self.model_dir)
        if self.best_score is None or total >= self.best_score:
            self.best_score = total
            return total
        return None

    def __getstate__(self) -> dict:
        keras.models.save_model(
            self.hist_model, f"{self.model_dir}/hist_model-{self.worker_id}.h5", save_format="h5"
        )
        keras.models.save_model(
            self.rt_model, f"{self.model_dir}/rt_model-{self.worker_id}.h5", save_format="h5"
        )
        return {
            "best_score": self.best_score,
            "explore": self.explore,
            "fitted": self.fitted,
            "avg_total": self.avg_total,
            "std_total": self.std_total,
        }

    def __setstate__(self, state: dict):
        self.hist_model = keras.models.load_model(f"{self.model_dir}/hist_model.h5")
        self.rt_model = keras.models.load_model(f"{self.model_dir}/rt_model.h5")
        self.best_score = state["best_score"]
        self.explore = state["explore"]
        self.fitted = state["fitted"]
        self.avg_total = state.get("avg_total")
        self.std_total = state.get("std_total")

    def save_checkpoint(self, checkpoint_dir: str = None) -> str:
        if checkpoint_dir is None:
            checkpoint_dir = self.model_dir
        checkpoint_path = f"{checkpoint_dir}/agent-{self.worker_id}.dat"
        pickle.dump(self.__getstate__(), open(checkpoint_path, "wb"))
        self.model_changed = True
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str):
        extra_data = pickle.load(open(checkpoint_path, "rb"))
        self.__setstate__(extra_data)
