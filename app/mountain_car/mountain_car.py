def mountain_car_potential_function(state):
    height = 0
    speed = 0
    g = 9.82
    return (height * g) + (speed * speed / 2)
