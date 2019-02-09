scaling_factor = 1


def potential_function(state):
    x_pos = state[0][0]
    speed = state[0][1]
    # print(f'speed: {speed:4.2f}, xpos: {x_pos:4.2f}')
    # return ((speed * speed / 2) + x_pos * 5) * scaling_factor
    return x_pos * 100 + speed * speed * 10
