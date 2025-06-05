from gym_duckietown.simulator import Simulator

env = Simulator(
    seed=123,
    map_name="loop_empty",
    max_steps=500001,
    domain_rand=0,
    camera_width=640,
    camera_height=480,
    accept_start_angle_deg=4,
    full_transparency=True,
    distortion=True,
)

obs = env.reset()
env.render()
input("Presiona Enter para cerrar la ventana...")
env.close()
