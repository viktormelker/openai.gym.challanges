import gym
import numpy as np
import tensorflow as tf

env = gym.make('CartPole-v0')
tf.reset_default_graph()

#These lines establish the feed-forward part of the network used to choose actions
inputs1 = tf.placeholder(shape = [1, 4], dtype = tf.float32)
W = tf.Variable(tf.random_uniform([4, 2], 0, 0.01))
Qout = tf.matmul(inputs1, W)
predict = tf.argmax(Qout, 1)

#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape = [1, 2], dtype = tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate = 0.15)
updateModel = trainer.minimize(loss)

init = tf.initialize_all_variables()

# Set learning parameters
randomness_probability = 0.10
num_episodes = 300
max_time = 300

# create lists to contain total rewards and steps per episode
jList = []
rList = []
with tf.Session() as sess:
    sess.run(init)
    for i_episode in range(num_episodes):
        observation = env.reset()
        rAll = 0
        done = False
        current_step = 0

        while current_step < max_time:
            current_step += 1

            input = np.reshape(observation, (1, 4))
            # Choose an action by greedily (with randomness_probability chance
            # of random action) from the Q-network
            action, allQ = sess.run([predict, Qout],
                                    feed_dict={inputs1: input})
            if np.random.rand(1) < randomness_probability:
                action[0] = env.action_space.sample()

            # Get new state and reward from environment
            observation, reward, done, _ = env.step(action[0])
            env.render()

            if done:
                print("Episode finished after {} timesteps".format(
                    current_step + 1))
                # Reduce chance of random action as we train the model.
                randomness_probability = 1./((current_step/25) + 10)
                break
