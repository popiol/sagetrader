import math


class Supervised:
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def get_action(self, x, is_hist):
        return self.action_space.sample()


class TriangleSupervised(Supervised):
    def get_action(self, x, is_hist):
        if x is None:
            confidence = 0
        else:
            p = [self.env.relative_price_decode(a[0]) for a in x]
            confidence = 0
            confidence += math.exp(-abs(p[-1] - 0.01) / 0.01)
            confidence += math.exp(-abs(p[-2] + 0.06) / 0.06)
            confidence += math.exp(-abs(p[-3] - 0.05) / 0.05)
            confidence += math.exp(-abs(p[-6] - 0.3) / 0.3)
            confidence = min(1, max(0, confidence / 4))
        buy_price = self.env.relative_price_encode(0)
        sell_price = self.env.relative_price_encode(0)
        return [confidence, buy_price, sell_price]
