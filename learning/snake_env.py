import gym
from game.snake import Game
from loguru import logger
from gym import spaces


class SnakeEnv(gym.Env):

    metadata = {'render.modes': ['human', 'ansi']}

    game = Game()

    obs_size = 0

    def __init__(self):

        self.obs_size = self.game.x_grid_size**2+4
        #self.obs_size = 7
        self.game.init()
        self.observation_space = spaces.Box(low=0, high=50, shape=(self.obs_size,))
        self.action_space = spaces.Discrete(3)

    def step(self, action):

        # print(action)

        # actions: 0 - continue, 1 - turn left, 2 - turn right
        #print("ACTION:", action.item())
        act = action.item()
        if act == 1:
            self.game.turn_left()
        if act == 2:
            self.game.turn_right()

        if self.game.game_over_state:
            self.reset()

        reward, done = self.game.game_loop()

        obs = self.game.get_observation()

        #print(reward)

        return obs, reward, done, ""

    def reset(self):

        # print("GAME IS OVER")

        self.game.game_over()
        return self.game.get_observation()

    def render(self, mode='human', close=False):
        pass
