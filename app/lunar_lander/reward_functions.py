from math import sqrt


def potential_function(state):
    pos_x = state[0][0]
    pos_y = state[0][1]
    lander_angle = state[0][4]
    angular_velocity = state[0][5]
    # print(f'pos (x,y) = ({pos_x:4.2f}, {pos_y:4.2f}), angle: {lander_angle:5.3f}')
    return (
        -abs(pos_x)
        - sqrt(pos_x ** 2 + pos_y ** 2)
        - abs(lander_angle)
        - abs(angular_velocity)
    )
