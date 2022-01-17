from simulator import StocksHistSimulator
import numpy as np


class _TestStocksHistSimulator:
    def test_check_closed(self):
        sim = StocksHistSimulator(max_steps=1, n_comps=1)
        assert not sim.file.closed
        sim.step([0, [0, 0, 0]])
        assert sim.file.closed

    def test_check_state_reward_format(self):
        sim = StocksHistSimulator(max_steps=1, n_comps=1, max_quotes=10)
        state, reward, done, info = sim.step([0, [0, 0, 0]])
        assert len(state) == 10
        for x in state:
            assert type(x) == float or type(x.item()) == float
        assert np.shape(state) == np.shape(sim.observation_space.sample())
        assert type(reward) == float
        assert type(done) == bool
        assert done
        assert info == {}

    def test_state_sequence(self):
        sim = StocksHistSimulator(max_steps=20, n_comps=1, max_quotes=10)
        for _ in range(19):
            action = sim.action_space.sample()
            state, reward, done, info = sim.step(action)
            assert not done
        comp = list(sim.prices)[0]
        assert np.isclose(sim.relative_price_decode(state[-1]), sim.prices[comp][-1] / sim.prices[comp][-2] - 1)
        sim.file.close()

    def test_order_expire(self):
        sim = StocksHistSimulator(max_steps=20, n_comps=1, max_quotes=10)
        comp = list(sim.prices)[0]
        for _ in range(10):
            dt1 = sim.timestamps[comp][-1].split()[0]
            hour = sim.timestamps[comp][-1].split()[1].split(":")[0]
            if int(hour) <= 10:
                break
            sim.step([[0, 0, 0]])
        action = [[0.5, 0, 0]]
        for _ in range(10):
            state, reward, done, info = sim.step(action)
            dt2 = sim.timestamps[comp][-1].split()[0]
            if dt2 == dt1:
                assert len(sim.orders) == 1
                assert sim.orders[comp]["buy"]
                limit = sim.orders[comp]["limit"]
                n_shares = sim.orders[comp]["n_shares"]
                assert sim.cash / 2 - limit <= n_shares * limit <= sim.cash / 2 + limit
            else:
                assert len(sim.orders) == 0
            action = [[0, 0, 0]]
        sim.file.close()
