"""
This is an example on how to use the two player Wimblepong environment
with two SimpleAIs playing against each other
"""
import matplotlib.pyplot as plt
from random import randint
import pickle
import gym
import numpy as np
import argparse
import wimblepong
from PIL import Image
from ac_agent import Agent
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--fps", type=int, help="FPS for rendering", default=30)
parser.add_argument("--scale", type=int, help="Scale of the rendered game", default=1)
args = parser.parse_args()

# Make the environment
env = gym.make("WimblepongVisualMultiplayer-v0")
env.unwrapped.scale = args.scale
env.unwrapped.fps = args.fps
# Number of episodes/games to play
episodes = 2000000

# Define the player IDs for both SimpleAI agents
learner_id = 1
teacher_id = 3 - learner_id

learner_validator_id = 1
validator_id = 3 - learner_validator_id

learner = Agent()
teacher = Agent()

filename = '../local_files/trained_nets/0.712_winrate0.87_valwinrate_self_play_with_validation_with_upgrades_new.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = torch.load(filename, map_location=device)
learner.policy.load_state_dict(weights, strict=False) #Load weights for learner
teacher.policy.load_state_dict(weights, strict=False) #Load weights for teacher
teacher.policy.eval()
teacher.test = True

learner_validator = Agent()
validator = wimblepong.SimpleAi(env, validator_id)


# Set the names for both SimpleAIs
env.set_names(learner.get_name(), teacher.get_name())

def validation_run(env,n_games=100):
    learner_validator.policy.load_state_dict(learner.policy.state_dict())
    learner_validator.policy.eval()
    learner_validator.test = True
    env.set_names(learner_validator.get_name(), validator.get_name())

    win1 = 0

    for j in range(0, n_games):
        done = False
        ob1, ob2 = env.reset()
        action_dist = [0,0,0]
        while not done:
            action1,_,_,_ = learner_validator.get_action(ob1)
            action2 = validator.get_action(ob2)
            action_dist[action1] += 1

            (ob1, ob2), (rew1, rew2), done, info = env.step((action1, action2))

            if not args.headless:
                env.render()

            if rew1 == 10:
                win1 += 1
        learner_validator.reset()

        print("Validation episode over:",j,"WR:",str(win1/(j+1)),"wins:", win1, "action dist", action_dist)
    
    return win1/n_games

run_title = "self_play_with_validation_with_upgrades_new2"
highest_running_winrate = 0
highest_validation_winrate = 0
save_every = 10000
validate_every = 3000
validation_winrate = 0
scores = [0 for _ in range(1000)]
win1 = 0
frames_seen = 0
opponent = "teacher"
change_opp_every = 10
for episode_number in range(0,episodes):
    reward_sum, timesteps = 0, 0
    done = False
    ob1, ob2 = env.reset()
    action_dist = [0,0,0]
    while not done:
        frames_seen += 1
        # Get the actions from both SimpleAIs
        action1, action_prob, entropy, state_value = learner.get_action(ob1)
        action_dist[action1] += 1
        if opponent == "teacher":
            action2,_,_,_ = teacher.get_action(ob2)
        else:
            action2 = validator.get_action(ob2)
        # Step the environment and get the rewards and new observations
        (ob1, ob2), (rew1, rew2), done, info = env.step((action1, action2))
        if rew1 == 10:
            win1 += 1

        learner.store_outcome(state_value, rew1, action_prob, entropy)
            
        # Store total episode reward
        reward_sum += rew1
        timesteps += 1

        if not args.headless:
            env.render()

    if rew1 == 10:
        scores.append(1)
        scores.pop(0)
    else:
        scores.append(0)
        scores.pop(0)
    
    run_avg = np.mean(np.array(scores))
    if  (run_avg - 0.01)  > highest_running_winrate:
        highest_running_winrate = run_avg
        torch.save(learner.policy.state_dict(), "../local_files/trained_nets/"+str(highest_running_winrate)+"_winrate"+str(highest_validation_winrate)+"_valwinrate_"+run_title+".pth")
        teacher.policy.load_state_dict(learner.policy.state_dict())
        teacher.policy.eval()
        teacher.test = True
        print("Teacher upgraded")

    learner.episode_finished()
    learner.reset()
    teacher.reset()

    if episode_number % change_opp_every == 0:
        if opponent == "teacher":
            print("Swithced to simpleAI")
            opponent = "simpleai"
        else:
            print("Swithced to teacher")
            opponent = "teacher"
            

    if episode_number % validate_every == 0:
        validation_winrate = validation_run(env)
        if validation_winrate > highest_validation_winrate:
            highest_validation_winrate = validation_winrate
            torch.save(learner.policy.state_dict(), "../local_files/trained_nets/"+str(highest_running_winrate)+"_winrate"+str(highest_validation_winrate)+"_valwinrate_"+run_title+".pth")
        env.set_names(learner.get_name(), teacher.get_name())
        print("Validation done")

    if episode_number  % save_every == 0:
        torch.save(learner.policy.state_dict(), "../local_files/trained_nets/"+str(highest_running_winrate)+"_winrate_"+str(highest_validation_winrate)+"_valwinrate_"+str(episode_number)+"_episodes_"+run_title+".pth")

    print("("+run_title+") Episode over:",str(episode_number),"WR:",str(win1/(episode_number+1)),"wins:", win1, "steps", str(timesteps) ,"frames seen:", frames_seen, "action dist", action_dist, "highest winrate", highest_running_winrate, "current winrate:", run_avg, "highest val winrate", highest_validation_winrate, "last val winrate", validation_winrate)
torch.save(learner.policy.state_dict(), "../local_files/trained_nets/"+str(highest_running_winrate)+"_winrate_at_end_"+run_title+"_.pth")