from app.policy import RandomPolicy


class TestRandomPolicy:
    state_size = 2
    action_size = 2

    policy = RandomPolicy(state_size=state_size, action_size=action_size)

    def test_get_action_works(self):
        action = self.policy.get_action(1, 1)

        assert action <= self.action_size
