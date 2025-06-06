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
from duckyPM.OdometryTracker import Odometria
from duckyPM.TrajectoryDrawer import TrajectoryDrawer
from duckyPM.IntersectionHandler import IntersectionHandler

class LaneFollowSimulator:
    def __init__(self, args):
        self.logger = setup_logger()
        self.logger.info("Inicializando simulador Duckietown con odometría y planificación")

        self.baseline = 0.08  # Distancia entre ruedas (m)
        self.frame_rate = 30  # Frecuencia de cuadros del entorno

        # Parámetros de simulación
        self.sample_count = args.sample_count  # Número de filas muestreadas para la ruta
        self.min_samples = args.min_samples
        
        # Inicalizar el objeto IntersectionHandler
        self.logger.info("Inicializando manejador de intersecciones")
        self.intersection_handler = IntersectionHandler(self)

        # Inicializar el objeto VisionProcessor
        self.logger.info("Inicializando procesador de visión")
        self.vision = VisionProcessor(self)

        # Inicializar el objeto Odometria
        self.logger.info("Inicializando odometría")
        self.odometria = Odometria(self)

        # Inicializar el objeto TrajectoryDrawer
        self.trajectory_drawer = TrajectoryDrawer(args.canvas_size, args.scale)

        self.env = self._initialize_env(args)

        self._init_simulation_parameters(args)    

        # Parámetros de visualización
        self.canvas_size = args.canvas_size  # Tamaño del lienzo de la trayectoria
        self.scale = args.scale  # Escala para convertir las unidades a píxeles

        # Inicializar lienzo para la trayectoria
        self.traj_canvas = np.ones((self.canvas_size, self.canvas_size, 3), dtype=np.uint8) * 255
        self.canvas_center = self.canvas_size // 2  # Centro del lienzo

        # Inicializar lista de trayectoria
        self.traj = []  # Lista para almacenar la trayectoria del robot (puntos (x, y))

        # Calcular el tiempo de cuadro
        self.dt = 1.0 / self.frame_rate if self.frame_rate > 0 else 0.1  # Tiempo entre cuadros
        self.logger.info("Simulador Duckietown inicializado correctamente.")

    def _initialize_env(self, args):
        """
        Inicializa el entorno de Duckietown con los parámetros especificados.
        Args:
            args: Argumentos de línea de comandos con parámetros del entorno.
        Returns:
            env: Instancia del entorno de Duckietown configurada.
        """
        try: 
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
            self.logger.info("Entorno Duckietown inicializado correctamente.")
            return env
        except Exception as e:
            self.logger.error(f"Error al inicializar el entorno Duckietown: {e}")
            return None  # Asegúrate de devolver None si hay algún error


    def _init_simulation_parameters(self, args):
        """Inicializa los parámetros de simulación y visión.
        Args:
            args: Argumentos de línea de comandos con parámetros de simulación.
        """
        self.logger.info("Inicializando parámetros de simulación y visión")
        # Parámetros de visión
        self.lower_bound = np.array([20, 100, 100])  # Valores mínimos para el amarillo
        self.upper_bound = np.array([30, 255, 255])  # Valores máximos para el amarillo
        self.speed = args.speed  # velocidad lineal base (m/s)
        self.lookahead_dist = args.lookahead_dist  # fracción a lo largo de la ruta planificada
        self.offset_ratio = args.offset_ratio  # relación de desplazamiento para conducción en el carril derecho
        self.kernel = np.ones((args.kernel_size, args.kernel_size), np.uint8)  # tamaño del núcleo morfológico

        # Intersecciones
        self.red_lower1 = np.array([0, 100, 100])  # Rango inferior del rojo
        self.red_upper1 = np.array([10, 255, 255])  # Rango superior del rojo
        self.red_lower2 = np.array([160, 100, 100])  # Rango inferior del rojo (cerca del 180)
        self.red_upper2 = np.array([180, 255, 255])  # Rango superior del rojo (cerca del 180)

#       # Parámetros de manejo de intersecciones
        self.red_threshold = args.red_threshold
        self.intersection_speed = args.intersection_speed
        self.turn_omega = args.turn_omega

        # Ruta navegación
        self.route = []  # Ruta predefinida cargada desde un archivo JSON
        self.route_index = 0  # Índice actual en la ruta
        self._load_route_from_file(args.route_file)

        # PID y otros parámetros de control
        self.Kp = args.kp  # Ganancia proporcional para el control PID
        self.max_omega = args.max_omega  # Máxima velocidad angular (rad/s)
        self.alpha = args.alpha  # Factor de suavizado para omega
        self.prev_omega = 0.0  # Omega anterior para suavizado

        #Máquina de estados mejorada para intersecciones
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

    def _load_route_from_file(self, route_file):
        """
        Carga una ruta predefinida desde un archivo JSON.
        Args:
            route_file: Ruta al archivo JSON que contiene la ruta.
        """
        if route_file and os.path.exists(route_file):
            try:
                with open(route_file, 'r') as f:
                    self.route = json.load(f)
                self.logger.info(f"Ruta cargada: {self.route}")
            except Exception as e:
                self.logger.warning(f"Error al cargar el archivo de ruta: {e}")
    
    def run(self):
        obs = self.env.reset()
        self.env.render()

        try:
            while True:
                # Procesar el frame
                speed, omega, vis = self.vision.process_frame(obs)

                # Asegurarse de que los valores de speed y omega sean válidos
                if speed is None or omega is None:
                    self.logger.error("Valores inválidos para speed y omega. Abortando ejecución.")
                    break  # Salir si los valores son inválidos

                # Actualizar odometría
                self.odometria.update_odometry(speed, omega)

                # Obtener posición actual
                x, y, theta = self.odometria.get_position()

                # Actualizar la trayectoria con la nueva posición
                self.trajectory_drawer.update_trajectory(x, y)

                # Dibujar la trayectoria actualizada
                self.trajectory_drawer.draw_trajectory()
                cv2.imshow("Trayectoria", self.trajectory_drawer.get_canvas())

                # Mostrar la detección
                cv2.imshow("Detección", cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
                cv2.waitKey(1)

                # Acción para el simulador
                action = np.array([speed, float(omega)])
                obs, _, done, _ = self.env.step(action)

                # Verificar si el episodio ha terminado y reiniciarlo si es necesario
                if done:
                    obs = self.safe_reset()
                    self.logger.info("Episodio reiniciado")

                # Renderizar el entorno
                self.env.render()

                # Esperar el tiempo de cuadro
                time.sleep(self.dt)

        except KeyboardInterrupt:
            pass

        finally:
            self.env.close()
            cv2.destroyAllWindows()
            
    def safe_reset(self):
        """
        Reinicia el entorno de Duckietown de manera segura.
        Maneja el caso en que el entorno devuelve una tupla.
        Returns:
            obs: Observación del entorno después del reinicio.
        """
        out = self.env.reset()
        if isinstance(out, tuple):
            return out[0]
        return out

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
    parser.add_argument('--lookahead-dist', default=0.5, type=float, help='Fracción a lo largo de la ruta planificada')
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

    # Crear y ejecutar el simulador
    sim = LaneFollowSimulator(args)
    sim.run()
