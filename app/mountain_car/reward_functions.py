scaling_factor = 1


def potential_function(state):
    x_pos = state[0][0]
    speed = state[0][1]
    return x_pos * 100 + speed * speed * 50000
