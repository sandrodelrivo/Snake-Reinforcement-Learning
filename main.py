from datetime import datetime
from pathlib import Path
import torch

from gym.wrappers import FrameStack

from learning.snake_agent import SnakeAgent
from learning.snake_env import SnakeEnv
from learning.snake_wrapper import SkipFrame
from logger import MetricLogger
import datetime


def main():
    save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)

    env = SnakeEnv()
    env.mode = 'ansi'

    # init environment
    env.reset()

    #print(env.obs_size)
    snake = SnakeAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n, save_dir=save_dir)

    logger = MetricLogger(save_dir)

    # prepare input

    episodes = 1000000
    for e in range(episodes):

        state = env.reset()

        # Play the game!
        while True:

            # Run agent on the state
            dist, action = snake.act(state)

            # Agent performs action
            next_state, reward, done, info = env.step(action)
            #print("NEXT STATE:", next_state)

            # Remember
            obs = (state, next_state, action, reward, done, dist)

            #print(obs)

            # Learn
            act_loss, crit_loss = snake.learn(obs)

            # Logging
            #print("Reward:", reward)

            logger.log_step(reward, act_loss, crit_loss)

            # Update state
            state = next_state

            # Check if end of game
            if done:
                break

        logger.log_episode()

        if e % 100 == 0:
            logger.record(episode=e, epsilon=0, step=snake.curr_step)
            pass

    return 0


main()
