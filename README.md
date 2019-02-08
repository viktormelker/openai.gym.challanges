# openai.gym.challanges
This repository contains attempts to solve various challanges from https://gym.openai.com/

## Environment
Use python 3.6, because tensorflow does not work with 3.7

## Launch simulation
I.e.
```sh
python -m app.lunar_lander.lunar_lander_v2_keras
```

## Running in docker
To avoid having to set up `keras` and `tensorflow` with dependencies the
simulations can be run in docker (without GUI).

To enter a shell in a docker image perform the following steps:

1. `docker build . -t keras_image`

2. `docker run -it keras_image bash`

After this the simulations can be launched as normally.