scaling_factor = 0.1


def potential_function(state):
    height = 0
    speed = 0
    g = 9.82
    return (height * g) + (speed * speed / 2) * scaling_factor
