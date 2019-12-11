import wimblepong
import gym
from agent import Agent
import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

import gym
from gym import logger as gymlogger
from gym.wrappers import Monitor
gymlogger.set_level(40) #error only

import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline
import math
import glob
import io
import base64
from IPython.display import HTML
from IPython import display as ipythondisplay
from pyvirtualdisplay import Display
#display = Display(visible=0, size=(1400, 900))
#display.start()
"""
Utility functions to enable video recording of gym environment and displaying it
To enable video, just do "env = wrap_env(env)""
"""

def show_video():
  mp4list = glob.glob('video/*.mp4')
  if len(mp4list) > 0:
    mp4 = mp4list[0]
    video = io.open(mp4, 'r+b').read()
    encoded = base64.b64encode(video)
    ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
  else: 
    print("Could not find video")
    

def wrap_env(env):
  env = Monitor(env, './video', force=True, write_upon_reset=True)
  return env



parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--fps", type=int, help="FPS for rendering", default=30)
parser.add_argument("--scale", type=int, help="Scale of the rendered game", default=1)
parser.add_argument("--filename", type=str, help="Weights to test", default=None)
args = parser.parse_args()

# Make the environment
env = gym.make("WimblepongVisualSimpleAI-v0")
env.unwrapped.scale = args.scale
env.unwrapped.fps = args.fps
# Number of episodes/games to play


player = Agent()

env.set_names("Policy gradient")
player.load_model()
#env = wrap_env(env)
wins = 0
for i in range(100):
    done = False
    observation = env.reset()
    while not done:
        action = player.get_action(observation)
        observation, reward, done, info = env.step(action)
        
        if reward == 10:
            wins += 1
        if i > 50:
            env.render()

    player.reset()
    print("Episode over:", i, "wins:", wins, "winrate:", wins/(i+1))

env.save_video()