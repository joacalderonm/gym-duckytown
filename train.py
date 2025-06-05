# import numpy as np
# import math
# import random
# from gym_duckietown.simulator import Simulator

# # Crear entorno simulado
# env = Simulator(
#     seed=123, 
#     map_name="loop_empty", 
#     max_steps=200,
#     domain_rand=False, 
#     camera_width=640, 
#     camera_height=480,
#     accept_start_angle_deg=4, 
#     full_transparency=True
# )

# # Definir acciones posibles: avanzar, girar derecha, girar izquierda
# actions = {
#     0: [0.3, 0.0],    # avanzar más lento
#     1: [0.3, 0.5],    # giro suave derecha
#     2: [0.3, -0.5]    # giro suave izquierda
# }

# # Q-table: cada celda del entorno será un estado
# grid_size = env.grid_width * env.grid_height
# q_table = np.zeros((env.grid_width * env.grid_height * 8, len(actions)))

# # Parámetros del algoritmo
# episodes = 50
# alpha = 0.1      # tasa de aprendizaje
# gamma = 0.9      # factor de descuento
# epsilon = 0.2    # exploración

# def get_state(env):
#     x = int(env.cur_pos[0])
#     y = int(env.cur_pos[2])
    
#     # Convertir orientación de radianes a grados y a dirección discreta
#     theta_deg = math.degrees(env.cur_angle) % 360
#     theta = int(theta_deg // 45)  # 8 direcciones: 0, 1, ..., 7

#     # Validar que x, y estén dentro de la grilla
#     if x < 0 or x >= env.grid_width or y < 0 or y >= env.grid_height:
#         return 0  # fallback a un estado base válido

#     return (y * env.grid_width + x) * 8 + theta

# # Entrenamiento
# for episode in range(episodes):
#     env.reset()
#     done = False
#     state = get_state(env)
#     total_reward = 0

#     while not done:
#         # Render solo en los últimos 10 episodios
#         env.render(mode="top_down")

#         if random.uniform(0, 1) < epsilon:
#             action_idx = random.choice(list(actions.keys()))
#         else:
#             action_idx = np.argmax(q_table[state])

#         obs, reward, done, _ = env.step(actions[action_idx])
#         next_state = get_state(env)

#         # Q-learning update
#         old_value = q_table[state, action_idx]
#         next_max = np.max(q_table[next_state])
#         q_table[state, action_idx] = old_value + alpha * (reward + gamma * next_max - old_value)

#         state = next_state
#         total_reward += reward

#     print(f"Ep {episode + 1}: Recompensa total = {total_reward:.2f}")


# # Guardar Q-table
# np.save("q_table.npy", q_table)

# env.close()

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
parser.add_argument('--map-name', default='udem1') # check maps into gym_duckietown/maps
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

env.reset()
env.render()

actions = {
    0: [0.4, 0.0],     # avanzar recto
    1: [0.4, -0.5],    # girar derecha leve
    2: [0.4, +0.5],    # girar izquierda leve
}

# Crear tabla Q
grid_size = env.grid_width * env.grid_height
q_table = np.zeros((grid_size, len(actions)))

# Parámetros Q-learning
episodes = 500
alpha = 0.2   # tasa de aprendizaje
gamma = 0.8    # factor de descuento
epsilon = 0.1  # exploración

def get_state(env):
    z = env.cur_pos[2]
    state = int(z * 5)  # por ejemplo: 5 estados por metro
    return min(state, q_table.shape[0] - 1)  # evita desbordarse

# Entrenamiento
for ep in range(episodes):
    obs = env.reset()
    env.unwrapped.cur_pos = [2.5, 0.0, 1.0]  # posición inicial
    env.unwrapped.cur_angle = 0.0


    state = get_state(env)
    total_reward = 0
    done = False

    while not done:
        env.render(mode="top_down")

        if random.uniform(0, 1) < epsilon:
            action_idx = random.choice(list(actions.keys()))
        else:
            action_idx = np.argmax(q_table[state])

        action = actions[action_idx]
        obs, reward, done, _ = env.step(action)

        try:
            lane_pos = env.unwrapped.get_lane_pos(env.unwrapped.cur_pos, env.unwrapped.cur_angle)
            dist = abs(lane_pos.dist)
            reward -= dist * 2

            if abs(action[1]) > 0.3:
                reward -= 0.05

            if dist < 0.05:
                reward += 0.2
        except:
            reward -= 5  # se salió o no se pudo calcular la distancia

        total_reward += reward

        next_state = get_state(env)

        q_table[state, action_idx] += alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state, action_idx]
        )

        state = next_state

    print(f"Episodio {ep+1}: Recompensa total = {total_reward:.2f}")


env.close()
np.save("q_table_straight2.npy", q_table)
