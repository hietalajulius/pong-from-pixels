"""
This is an example on how to use the two player Wimblepong environment with one
agent and the SimpleAI
"""
import matplotlib.pyplot as plt
from random import randint
import pickle
import gym
import numpy as np
import argparse
import wimblepong
from policy_gradient_agent import Agent as PolicyAgent
from dqn_agent import Agent as DqnAgent
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--housekeeping", action="store_true", help="Plot, player and ball positions and velocities at the end of each episode")
parser.add_argument("--fps", type=int, help="FPS for rendering", default=30)
parser.add_argument("--scale", type=int, help="Scale of the rendered game", default=1)
args = parser.parse_args()

# Make the environment
env = gym.make("WimblepongVisualSimpleAI-v0")
env.unwrapped.scale = args.scale
env.unwrapped.fps = args.fps

# Number of episodes/games to play



# Define the player
player_id = 1
# Set up the player here. We used the SimpleAI that does not take actions for now
player = DqnAgent()#wimblepong.SimpleAi(env, player_id)

TARGET_UPDATE = 10000

# Housekeeping
states = []
win1 = 0
env.set_names(player.get_name())
games = 0

#for i in range(0,episodes):
if player.train:
    #Test network saving
    torch.save(player.policy_net.state_dict(), "../local_files/trained_nets/net_at_start.pth")
    while player.frames_seen < player.target_frames:
        done = False
        games += 1
        observation = env.reset()
        action = 0
        steps = 0
        while not done:
            steps += 1
            previous_observation = observation
            action = player.get_action(observation)

            observation, reward, done, info = env.step(action)
            if reward < 0:
                clipped_reward = -1
            elif reward > 0:
                clipped_reward = 1
            else:
                clipped_reward = 0

            player.store_transition(action, clipped_reward, done)
            player.update_network()
            

            if args.housekeeping:
                states.append(observation)

            if player.frames_seen % TARGET_UPDATE == 0:
                player.update_target_network()
                
            # Count the wins
            if reward == 10:
                win1 += 1

            if player.frames_seen % 100000 == 0:
                torch.save(player.policy_net.state_dict(), "../local_files/trained_nets/net_at_"+ str(games) +"_games.pth")
            if not args.headless:
                env.render()
            if done:
                observation= env.reset()
                plt.close()  # Hides game window
                if args.housekeeping:
                    plt.plot(states)
                    plt.legend(["Player", "Opponent", "Ball X", "Ball Y", "Ball vx", "Ball vy"])
                    plt.show()
                    states.clear()
                print("episode {} over. Broken WR: {:.3f}".format(games, win1/(games+1)), "wins:", win1, "steps:", steps, "frames seen:", player.frames_seen, "epsilon:", player.epsilon)
                if games % 5 == 4:
                    env.switch_sides()
                
                player.reset()

    torch.save(player.policy_net.state_dict(), "../local_files/trained_nets/net_at_end.pth")
else:
    player.load_model()
    for games in range(100):
        done = False
        observation = env.reset()
        action = 0
        steps = 0
        while not done:
            steps += 1
            action = player.get_action(observation)
            observation, reward, done, info = env.step(action)

            # Count the wins
            if reward == 10:
                win1 += 1

            if not args.headless:
                env.render()
            if done:
                observation= env.reset()
                plt.close()  # Hides game window
                print("episode {} over. Broken WR: {:.3f}".format(games, win1/(games+1)), "wins:", win1, "steps:", steps, "frames seen:", player.frames_seen, "epsilon:", player.epsilon)
                if games % 5 == 4:
                    env.switch_sides()
                
