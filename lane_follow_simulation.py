#!/usr/bin/env python3
"""
Simulador de seguimiento de carriles de Duckiebot con planificación de rutas e integración de odometría a partir de comandos de ruedas.
Rastrea la pose del robot integrando cinemática de tracción diferencial y dibuja tanto la detección como la trayectoria odométrica.
El origen del lienzo de trayectoria ahora está en el centro.
"""
import cv2
import numpy as np
import argparse
import time
import gym
import gym_duckietown
import json
import os
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import UndistortWrapper
from duckyPM.utils.logger import setup_logger
from duckyPM.VisionProcessor import VisionProcessor


class LaneFollowSimulator:
    def __init__(self, args):
        self.logger = setup_logger()
        self.logger.info("Inicializando simulador Duckietown con odometría y planificación")
        self.vision = VisionProcessor(self)

        # Límites HSV para la marca de carril amarillo - ajustados para evitar confusión con el pasto
        self.lower_bound = np.array([20, 100, 100])  # Valores mínimos aumentados
        self.upper_bound = np.array([35, 255, 255])  # Rango de tono reducido para evitar el verde

        # Parámetros de conducción
        self.speed = args.speed                # velocidad lineal (m/s)
        self.lookahead_dist = args.lookahead_dist  # fracción a lo largo de la ruta planificada

        # Relación de desplazamiento para conducción en el carril derecho
        self.offset_ratio = args.offset_ratio

        # Parámetros de manejo de intersecciones
        self.red_lower1 = np.array([0, 100, 100])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([160, 100, 100])
        self.red_upper2 = np.array([180, 255, 255])
        self.red_threshold = args.red_threshold
        self.intersection_speed = args.intersection_speed
        self.turn_omega = args.turn_omega
        
        # Máquina de estados mejorada para intersecciones
        self.state = 'FOLLOW'
        self.turn_action = None
        self.red_count_history = []
        self.red_history_size = 10
        self.red_peak_detected = False
        self.frames_after_peak = 0
        self.frames_to_wait = args.frames_to_wait  # Esperar este número de cuadros después del pico rojo antes de girar
        self.turn_start_frame = 0
        self.turn_duration = args.turn_duration  # Duración del giro en cuadros
        self.intersection_cooldown = 0
        self.cooldown_duration = args.cooldown_duration  # Cuadros a esperar antes de detectar otra intersección
        
        # Cargar ruta predefinida si está disponible
        self.route = []
        self.route_index = 0
        route_file = args.route_file
        if route_file and os.path.exists(route_file):
            try:
                with open(route_file, 'r') as f:
                    self.route = json.load(f)
                self.logger.info(f"Ruta cargada: {self.route}")
            except Exception as e:
                self.logger.warning(f"Error al cargar el archivo de ruta: {e}")

        # Ganancia PID
        self.Kp = args.kp

        # Suavizado y recorte de omega
        self.max_omega = args.max_omega
        self.alpha = args.alpha
        self.prev_omega = 0.0

        # Tamaño del núcleo morfológico
        self.kernel = np.ones((args.kernel_size, args.kernel_size), np.uint8)

        # Muestreo para planificación de rutas
        self.sample_count = args.sample_count
        self.min_samples = args.min_samples

        # Estado de odometría
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_theta = 0.0

        # Historial de trayectoria
        self.traj = []  # lista de (x, y)

        # Lienzo para la trayectoria
        self.canvas_size = args.canvas_size
        self.scale = args.scale  # píxeles por metro
        self.traj_canvas = np.ones((self.canvas_size, self.canvas_size, 3), dtype=np.uint8) * 255
        self.canvas_center = self.canvas_size // 2

        # Inicializar el entorno de Duckietown
        env = DuckietownEnv(
            seed=args.seed,
            map_name=args.map_name,
            draw_bbox=args.draw_bbox,
            draw_curve=args.draw_curve,
            domain_rand=args.domain_rand,
            frame_skip=args.frame_skip,
            distortion=args.distortion,
        )
        if args.distortion:
            env = UndistortWrapper(env)
        self.env = env

        # Parámetros cinemáticos del entorno
        self.baseline = getattr(self.env.unwrapped, 'baseline', 0.08)  # distancia entre ruedas (m)
        frame_rate = getattr(self.env.unwrapped, 'frame_rate', 30)
        self.dt = 1.0 / frame_rate if frame_rate > 0 else 0.1

    def safe_reset(self):
        out = self.env.reset()
        if isinstance(out, tuple):
            return out[0]
        return out

    def draw_trajectory(self):
        # Limpiar lienzo
        self.traj_canvas[:] = 255
        if len(self.traj) < 2:
            return
        # Convertir cada punto de odometría a coordenadas de píxeles con origen en el centro
        pts = []
        for x, y in self.traj:
            px = int(self.canvas_center + x * self.scale)
            py = int(self.canvas_center - y * self.scale)
            pts.append((px, py))
        # Dibujar líneas entre puntos consecutivos
        for i in range(1, len(pts)):
            cv2.line(self.traj_canvas, pts[i-1], pts[i], (255, 0, 0), 2)

    

    def run(self):
        obs = self.safe_reset()
        self.env.render()
        try:
            while True:
                # Seguimiento de carriles y obtener comando
                speed, omega, vis = self.vision.process_frame(obs)
                # Integración de odometría (tracción diferencial)
                v_l = speed - omega * (self.baseline / 2)
                v_r = speed + omega * (self.baseline / 2)
                d_l = v_l * self.dt
                d_r = v_r * self.dt
                d_center = (d_r + d_l) / 2.0
                d_theta = (d_r - d_l) / self.baseline
                self.odom_theta += d_theta
                self.odom_x += d_center * np.cos(self.odom_theta)
                self.odom_y += d_center * np.sin(self.odom_theta)
                self.traj.append((self.odom_x, self.odom_y))

                # Aplicar acción
                action = np.array([speed, float(omega)])
                out = self.env.step(action)
                if len(out) == 4:
                    obs, _, done, _ = out
                else:
                    obs = out[0]; done = out[2]

                # Dibujar trayectoria actualizada
                self.draw_trajectory()
                cv2.imshow("Trayectoria", self.traj_canvas)

                if done:
                    obs = self.safe_reset()
                    self.logger.info("Episodio reiniciado")

                # Mostrar detección
                cv2.imshow("Detección", cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
                cv2.waitKey(1)
                self.env.render()
                time.sleep(self.dt)
        except KeyboardInterrupt:
            pass
        finally:
            self.env.close()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Seguimiento de carriles de Duckiebot con integración de odometría')
    parser.add_argument('--map-name', default='udem1', help='Nombre del mapa de Duckietown')
    parser.add_argument('--distortion', action='store_true', help='Aplicar distorsión de cámara')
    parser.add_argument('--draw-curve', action='store_true', help='Dibujar curva de carril')
    parser.add_argument('--draw-bbox', action='store_true', help='Dibujar cuadros delimitadores')
    parser.add_argument('--domain-rand', action='store_true', help='Habilitar aleatorización de dominio')
    parser.add_argument('--frame-skip', default=1, type=int, help='Cuadros a omitir')
    parser.add_argument('--seed', default=1, type=int, help='Semilla aleatoria')
    parser.add_argument('--speed', default=0.2, type=float, help='Velocidad lineal base (m/s)')
    parser.add_argument('--kp', default=1.0, type=float, help='Ganancia proporcional para PID de dirección')
    parser.add_argument('--alpha', default=0.2, type=float, help='Factor de suavizado para omega')
    parser.add_argument('--max-omega', default=0.8, type=float, help='Máxima velocidad angular (rad/s)')
    parser.add_argument('--kernel-size', default=5, type=int, help='Tamaño del núcleo morfológico')
    parser.add_argument('--sample-count', default=20, type=int, help='Número de filas muestreadas para la ruta')
    parser.add_argument('--min-samples', default=3, type=int,
                        help='Número mínimo de puntos requeridos para ajuste polinómico')
    parser.add_argument('--lookahead-dist', default=0.5, type=float, help='Fracción de anticipación a lo largo de la ruta')
    parser.add_argument('--scale', default=50, type=int, help='Píxeles por metro para dibujar la trayectoria')
    parser.add_argument('--canvas-size', default=300, type=int, help='Tamaño del lienzo cuadrado de trayectoria')
    parser.add_argument('--cooldown-duration', default=20, type=int,
                        help='Número de cuadros a esperar antes de detectar otra intersección')
    parser.add_argument('--offset-ratio', default=0.4, type=float,
                        help='Fracción del ancho de la imagen para desplazar la detección a la izquierda (0-1)')
    parser.add_argument('--red-threshold', default=20000, type=int,
                        help='Umbral de conteo de píxeles para detectar franjas rojas de intersección')
    parser.add_argument('--intersection-speed', default=0.1, type=float,
                        help='Velocidad durante el cruce de intersección')
    parser.add_argument('--turn-omega', default=0.85, type=float,
                        help='Velocidad angular durante el giro en intersección')
    parser.add_argument('--frames-to-wait', default=20, type=int,
                        help='Número de cuadros a esperar después del pico rojo antes de girar')
    parser.add_argument('--turn-duration', default=17, type=int,
                        help='Duración del giro en cuadros')
    parser.add_argument('--route-file', default='route_4way.json', type=str,
                        help='Archivo JSON que contiene una ruta predefinida de giros')
    args = parser.parse_args()
    sim = LaneFollowSimulator(args)
    sim.run()