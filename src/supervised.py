import math


class Supervised:
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def get_action(self, x, is_hist):
        return self.action_space.sample()


class TriangleSupervised(Supervised):
    def get_action(self, x, is_hist):
        p = [self.env.relative_price_decode(a[0]) for a in x]
        confidence = 0
        confidence += math.exp(-abs(p[-1] - .01) / .01)
        confidence += math.exp(-abs(p[-2] + .06) / .06)
        confidence += math.exp(-abs(p[-3] - .05) / .05)
        confidence += math.exp(-abs(p[-6] - .3) / .3)
        confidence = min(1, max(0, confidence / 4))
        buy_price = self.env.relative_price_encode(0)
        sell_price = self.env.relative_price_encode(0)
        return [confidence, buy_price, sell_price]
