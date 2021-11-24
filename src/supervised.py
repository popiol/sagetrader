import tensorflow.keras as keras
from simulator import StocksRTSimulator


class Supervised():
    def __init__(self):
        self.hist_model = self.create_hist_model()
        self.rt_model = self.create_rt_model()
        self.price_ch_buckets = [-.27, -.09, -.03, -.01, .01, .03, .09, .27]
        self.hist_trainset = self.get_trainsets()

    def create_hist_model(self):
        inputs = keras.layers.Input(shape=(self.max_quotes, 5))
        l = inputs
        l = keras.layers.LSTM(self.max_quotes)(l)
        l = keras.layers.Dense(100, activation="relu")(l)
        l = keras.layers.Dense(10, activation="relu")(l)
        l = keras.layers.Dense(10, activation="relu")(l)
        l = keras.layers.Dense(10, activation="relu")(l)
        outputs = keras.layers.Dense(9, activation="softmax")(l)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Nadam(learning_rate=0.001),
            loss="categorical_crossentropy",
        )
        return model

    def create_rt_model(self):
        inputs = keras.layers.Input(shape=(self.max_quotes, 6))
        l = inputs
        l = keras.layers.LSTM(self.max_quotes)(l)
        l = keras.layers.Dense(100, activation="relu")(l)
        l = keras.layers.Dense(10, activation="relu")(l)
        l = keras.layers.Dense(10, activation="relu")(l)
        l = keras.layers.Dense(10, activation="relu")(l)
        outputs = keras.layers.Dense(9, activation="softmax")(l)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Nadam(learning_rate=0.001),
            loss="categorical_crossentropy",
        )
        return model

    def get_trainsets(self):
        simulator = StocksRTSimulator()

