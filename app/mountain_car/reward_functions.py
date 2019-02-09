scaling_factor = 0.1


def potential_function(state):
    x_pos = state[0][0]
    height = abs(x_pos - 0.5)
    speed = state[0][1]
    g = 9.82
    return ((height * g) + (speed * speed / 2) + x_pos) * scaling_factor
