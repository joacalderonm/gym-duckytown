import numpy as np
import sys
import argparse
import random
from gym_duckietown.envs import DuckietownEnv
import pyglet
from pyglet.window import key
import gym
import gym_duckietown
import time

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default='Duckietown_training') 
parser.add_argument('--map-name', default='straight_road') # check maps into gym_duckietown/maps
parser.add_argument('--distortion', default=False, action='store_true')
parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
parser.add_argument('--max-steps', default=300, type=int, help='number of steps to take in the environment')
parser.add_argument('--seed', default=1, type=int, help='seed')
args = parser.parse_args()

if args.env_name and args.env_name.find('Duckietown') != -1:
    env = DuckietownEnv(
        seed = args.seed,
        map_name = args.map_name,
        draw_curve = args.draw_curve,
        draw_bbox = args.draw_bbox,
        domain_rand = args.domain_rand,
        frame_skip = args.frame_skip,
        distortion = args.distortion,
    )
else:
    env = gym.make(args.env_name)


# Cargar modelo entrenado
q_table = np.load("q_table_straight2.npy")

actions = {
    0: [0.4, 0.0],     # avanzar recto
    1: [0.4, -0.5],    # girar derecha leve
    2: [0.4, +0.5],    # girar izquierda leve
}
def get_state(env):
    z = env.cur_pos[2]  # avance a lo largo de la carretera
    return min(int(z * 5), 49)  # por ejemplo, 50 estados posibles

# Simulación usando el modelo
env.reset()
done = False
state = get_state(env)

while not done:
    env.render()
    action_idx = np.argmax(q_table[state])
    obs, reward, done, _ = env.step(actions[action_idx])
    state = get_state(env)

env.close()
