import numpy as np
import common


class Supervised:
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def get_action(self, x, is_hist):
        return self.action_space.sample()


class TriangleSupervised(Supervised):
    def get_action(self, x, is_hist):
        p = [self.env.relative_price_decode(a[0]) for a in x]
        confidence = self.env.relative_price_encode((p[-4]) - (p[-1] + p[-2] + p[-3]))
        buy_price = self.env.relative_price_encode(0)
        sell_price = self.env.relative_price_encode(0)
        return [confidence, buy_price, sell_price]
