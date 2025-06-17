import cv2
import numpy as np
import time
import json
import os
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import UndistortWrapper

from ..vision.lane_detector import LaneDetector
from ..vision.intersection_detector import IntersectionDetector
from ..vision.obstacle_detector import ObstacleDetector
from ..control.motion_controller import MotionController
from ..utils.trajectory_visualizer import TrajectoryVisualizer

class LaneFollowSimulator:
    def __init__(self, args):
        # Inicializar componentes
        self.lane_detector = LaneDetector(
            kernel_size=args.kernel_size,
            sample_count=args.sample_count,
            min_samples=args.min_samples,
            offset_ratio=args.offset_ratio,
            dbscan_eps=args.dbscan_eps,
            dbscan_min_samples=args.dbscan_min_samples
        )
        
        self.intersection_detector = IntersectionDetector(
            kernel_size=args.kernel_size,
            red_threshold=args.red_threshold
        )
        
        self.obstacle_detector = ObstacleDetector(
            eps=args.dbscan_eps,
            min_samples=args.dbscan_min_samples
        )
        
        self.motion_controller = MotionController(
            baseline=0.08,  # Valor por defecto del entorno
            dt=1.0/30.0    # Valor por defecto del entorno
        )
        
        self.trajectory_visualizer = TrajectoryVisualizer(
            canvas_size=args.canvas_size,
            scale=args.scale
        )
        
        # Parámetros de control
        self.speed = args.speed
        self.lookahead_dist = args.lookahead_dist
        self.intersection_speed = args.intersection_speed
        self.turn_omega = args.turn_omega
        self.frames_to_wait = args.frames_to_wait
        self.turn_duration = args.turn_duration
        self.cooldown_duration = args.cooldown_duration
        
        # Parámetros de control PID
        self.kp = args.kp
        self.alpha = args.alpha
        self.max_omega = args.max_omega
        
        # Parámetros de evasión de obstáculos
        self.min_safe_distance = args.min_safe_distance
        self.obstacle_speed_factor = args.obstacle_speed_factor
        
        # Estado de la máquina de estados
        self.state = 'FOLLOW'
        self.turn_action = None
        self.turn_start_frame = 0
        
        # Cargar ruta predefinida
        self.route = []
        self.route_index = 0
        if args.route_file and os.path.exists(args.route_file):
            try:
                with open(args.route_file, 'r') as f:
                    self.route = json.load(f)
                print(f"Ruta cargada: {self.route}")
            except Exception as e:
                print(f"Error al cargar el archivo de ruta: {e}")
        
        # Inicializar entorno
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

    def safe_reset(self):
        out = self.env.reset()
        if isinstance(out, tuple):
            return out[0]
        return out

    def adjust_speed_for_obstacles(self, base_speed, cluster_centers, frame_shape):
        """
        Adjust speed based on detected obstacles.
        
        Args:
            base_speed: Base speed of the robot
            cluster_centers: List of cluster center coordinates
            frame_shape: Shape of the frame (height, width)
            
        Returns:
            float: Adjusted speed
        """
        if not cluster_centers:
            return base_speed
            
        # Convert frame coordinates to normalized coordinates (0-1)
        height, width = frame_shape[:2]
        normalized_centers = [(x/width, y/height) for x, y in cluster_centers]
        
        # Calculate minimum distance to any cluster center
        min_distance = float('inf')
        for center_x, center_y in normalized_centers:
            # Calculate distance to center of frame (0.5, 0.5)
            distance = np.sqrt((center_x - 0.5)**2 + (center_y - 0.5)**2)
            min_distance = min(min_distance, distance)
        
        # Adjust speed based on distance
        if min_distance < self.min_safe_distance:
            # Reduce speed proportionally to how close the obstacle is
            speed_factor = min_distance / self.min_safe_distance
            return base_speed * speed_factor * self.obstacle_speed_factor
        
        return base_speed

    def process_frame(self, frame):
        # Detección de intersección
        red_count, red_mask = self.intersection_detector.process_frame(frame)
        cv2.imshow("Máscara Roja", red_mask)
        
        # Máquina de estados para manejo de intersecciones
        if self.state == 'FOLLOW':
            if self.intersection_detector.is_intersection_detected():
                print("Línea roja detectada, preparándose para la intersección")
                self.state = 'APPROACHING'
                
                # Obtener siguiente giro
                if self.route:
                    self.turn_action = self.route[self.route_index % len(self.route)]
                    self.route_index += 1
                    print(f"Siguiendo ruta: giro {self.turn_action} (índice {self.route_index-1})")
                else:
                    self.turn_action = np.random.choice(['straight', 'right', 'left'])
                    print(f"Giro aleatorio: {self.turn_action}")
                
                self.intersection_detector.reset_peak_detection()
                
            # Seguimiento normal de carriles
            path_points, vis, masks = self.lane_detector.process_frame(frame, self.turn_action)
            
            # Mostrar máscaras de color
            cv2.imshow("Máscara Roja", masks['red'])
            cv2.imshow("Máscara Amarilla", masks['yellow'])
            cv2.imshow("Máscara Blanca", masks['white'])
            
            # Detectar obstáculos usando DBSCAN
            labels, cluster_centers, cluster_sizes = self.obstacle_detector.detect_clusters(path_points)
            
            # Visualizar clusters
            vis = self.obstacle_detector.visualize_clusters(vis, path_points, labels, cluster_centers)
            
            # Ajustar velocidad basada en obstáculos
            current_speed = self.adjust_speed_for_obstacles(
                self.speed, cluster_centers, frame.shape
            )
            
            # Calcular control
            omega = self.motion_controller.compute_control(
                path_points, self.lookahead_dist,
                self.kp, self.alpha, self.max_omega,
                int(frame.shape[1] * self.lane_detector.offset_ratio),
                frame.shape[1]
            )
            
            return current_speed, omega, vis
            
        elif self.state == 'APPROACHING':
            # Reducir velocidad al acercarse
            speed = self.intersection_speed * 1.5
            
            # Verificar si hemos pasado el pico de rojo
            if self.intersection_detector.frames_after_peak >= self.frames_to_wait:
                print(f"Comenzando giro {self.turn_action}")
                self.state = 'TURNING'
                self.turn_start_frame = 0
            
            # Continuar con seguimiento de carriles
            path_points, vis, masks = self.lane_detector.process_frame(frame, self.turn_action)
            
            # Mostrar máscaras de color
            cv2.imshow("Máscara Roja", masks['red'])
            cv2.imshow("Máscara Amarilla", masks['yellow'])
            cv2.imshow("Máscara Blanca", masks['white'])
            
            # Detectar obstáculos
            labels, cluster_centers, cluster_sizes = self.obstacle_detector.detect_clusters(path_points)
            vis = self.obstacle_detector.visualize_clusters(vis, path_points, labels, cluster_centers)
            
            # Ajustar velocidad
            current_speed = self.adjust_speed_for_obstacles(
                speed, cluster_centers, frame.shape
            )
            
            omega = self.motion_controller.compute_control(
                path_points, self.lookahead_dist,
                self.kp, self.alpha, self.max_omega,
                int(frame.shape[1] * self.lane_detector.offset_ratio),
                frame.shape[1]
            )
            
            return current_speed, omega * 0.7, vis
            
        elif self.state == 'TURNING':
            # Ejecutar giro
            speed = self.intersection_speed
            
            # Aplicar dirección según el giro
            if self.turn_action == 'left':
                turn_progress = min(1.0, self.turn_start_frame / 3.0)
                omega = self.turn_omega * 1.3 * turn_progress
            elif self.turn_action == 'right':
                turn_progress = min(1.0, self.turn_start_frame / 5.0)
                omega = -self.turn_omega * turn_progress
            else:  # recto
                omega = 0.1
            
            self.turn_start_frame += 1
            
            # Verificar fin del giro
            turn_duration_adjusted = self.turn_duration
            if self.turn_action == 'left':
                turn_duration_adjusted = int(self.turn_duration * 1.5)
                
            if self.turn_start_frame >= turn_duration_adjusted:
                print(f"Giro {self.turn_action} completado, regresando al seguimiento de carriles")
                self.state = 'FOLLOW'
                self.turn_action = None
                self.intersection_detector.set_cooldown(self.cooldown_duration)
            
            return speed, omega, frame

    def run(self):
        obs = self.safe_reset()
        self.env.render()
        
        try:
            while True:
                # Procesar frame y obtener comando
                speed, omega, vis = self.process_frame(obs)
                
                # Actualizar odometría
                self.motion_controller.update_odometry(speed, omega)
                
                # Aplicar acción
                action = np.array([speed, float(omega)])
                out = self.env.step(action)
                if len(out) == 4:
                    obs, _, done, _ = out
                else:
                    obs = out[0]
                    done = out[2]
                
                # Visualizar trayectoria
                traj_canvas = self.trajectory_visualizer.draw_trajectory(
                    self.motion_controller.get_trajectory()
                )
                cv2.imshow("Trayectoria", traj_canvas)
                
                if done:
                    obs = self.safe_reset()
                    print("Episodio reiniciado")
                
                # Mostrar detección
                cv2.imshow("Detección", cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
                cv2.waitKey(1)
                self.env.render()
                time.sleep(1.0/30.0)  # Valor por defecto del entorno
                
        except KeyboardInterrupt:
            pass
        finally:
            self.env.close()
            cv2.destroyAllWindows() 