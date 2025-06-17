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

class LaneFollowSimulator:
    def __init__(self, args):
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
                print(f"Ruta cargada: {self.route}")
            except Exception as e:
                print(f"Error al cargar el archivo de ruta: {e}")

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

    def process_frame(self, frame):
        # Preprocesar imagen para detección de carriles
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        h, w, _ = bgr.shape
        crop_top = h // 3
        crop = bgr[crop_top:, :]
        ch, cw = crop.shape[:2]

        # lienzo de visualización para este cuadro
        vis = crop.copy()

        blur = cv2.GaussianBlur(crop, (5, 5), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        # Detección de intersecciones a través de franjas rojas
        red_mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        red_mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, self.kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, self.kernel)
        red_count = cv2.countNonZero(red_mask)
        cv2.imshow("Máscara Roja", red_mask)
        
        # Actualizar historial de conteo rojo para detección de picos
        self.red_count_history.append(red_count)
        if len(self.red_count_history) > self.red_history_size:
            self.red_count_history.pop(0)
        
        # Disminuir contador de enfriamiento si está activo
        if self.intersection_cooldown > 0:
            self.intersection_cooldown -= 1
            if self.intersection_cooldown == 0:
                print("Enfriamiento de intersección terminado, listo para la próxima intersección")
        
        # Máquina de estados mejorada para manejo de intersecciones
        if self.state == 'FOLLOW':
            # Detectar línea roja y prepararse para la intersección (solo si no está en enfriamiento)
            if red_count > self.red_threshold and self.intersection_cooldown == 0:
                print("Línea roja detectada, preparándose para la intersección")
                self.state = 'APPROACHING'
                
                # Obtener el siguiente giro de la ruta si está disponible, de lo contrario usar elección aleatoria
                if self.route:
                    self.turn_action = self.route[self.route_index % len(self.route)]
                    self.route_index += 1
                    print(f"Siguiendo ruta: giro {self.turn_action} (índice {self.route_index-1})")
                else:
                    self.turn_action = np.random.choice(['straight', 'right', 'left'])
                    print(f"Giro aleatorio: {self.turn_action}")
                
                self.red_peak_detected = False
                self.frames_after_peak = 0
                
        elif self.state == 'APPROACHING':
            # Reducir velocidad al acercarse a la intersección
            speed = self.intersection_speed * 1.5  # Un poco más rápido que la velocidad de cruce
            
            # Verificar si hemos pasado el pico de rojo (lo que significa que estamos cruzando la línea)
            if len(self.red_count_history) >= 3:
                if not self.red_peak_detected and self.red_count_history[-1] < self.red_count_history[-2]:
                    # Hemos pasado el pico de rojo
                    self.red_peak_detected = True
                    print("Pico rojo detectado, esperando para cruzar la línea")
                
            # Después de detectar el pico, contar cuadros hasta que estemos más allá de la línea
            if self.red_peak_detected:
                self.frames_after_peak += 1
                
                # Una vez que hemos esperado suficientes cuadros después del pico, comenzar a girar
                if self.frames_after_peak >= self.frames_to_wait:
                    print(f"Comenzando giro {self.turn_action}")
                    self.state = 'TURNING'
                    self.turn_start_frame = 0
            
            # Continuar con el seguimiento de carriles mientras se acerca
            return speed, self.prev_omega * 0.7, vis  # Reducir dirección para acercarse más recto
            
        elif self.state == 'TURNING':
            # Ejecutar el giro después de cruzar la línea roja
            speed = self.intersection_speed
            
            # Aplicar dirección apropiada según la dirección del giro
            if self.turn_action == 'left':
                # Aumento gradual de la tasa de giro para giros a la izquierda
                # Usar una rampa más agresiva para giros a la izquierda
                turn_progress = min(1.0, self.turn_start_frame / 3.0)  # Más rápido que 5.0
                # Aumentar la velocidad angular para giros a la izquierda
                omega = self.turn_omega * 1.3 * turn_progress  # 30% más de velocidad angular
            elif self.turn_action == 'right':
                # Aumento gradual de la tasa de giro
                turn_progress = min(1.0, self.turn_start_frame / 5.0)
                omega = -self.turn_omega * turn_progress
            else:  # recto
                # Aplicar una pequeña corrección para mantener el robot centrado
                # Esto ayuda a contrarrestar la tendencia a desviarse hacia la derecha
                omega = 0.1  # Ligera corrección hacia la izquierda para mantener la trayectoria recta
            
            # Incrementar contador de cuadros de giro
            self.turn_start_frame += 1
            
            # Salir del estado de giro después de suficientes cuadros
            # Para giros a la izquierda, aumentar la duración del giro
            turn_duration_adjusted = self.turn_duration
            if self.turn_action == 'left':
                turn_duration_adjusted = int(self.turn_duration * 1.5)  # 50% más de tiempo para giros a la izquierda
                
            if self.turn_start_frame >= turn_duration_adjusted:
                print(f"Giro {self.turn_action} completado, regresando al seguimiento de carriles")
                self.state = 'FOLLOW'
                self.turn_action = None
                # Establecer enfriamiento para evitar la detección inmediata de la intersección
                self.intersection_cooldown = self.cooldown_duration
                print(f"Estableciendo enfriamiento de intersección por {self.cooldown_duration} cuadros")
            
            return speed, omega, vis

        # Detección de línea amarilla con filtrado mejorado para evitar el pasto
        mask = cv2.inRange(hsv, self.lower_bound, self.upper_bound)
        
        # Filtrar áreas que probablemente sean pasto (verde)
        # Crear una máscara para el color verde (pasto)
        green_lower = np.array([35, 50, 50])  # Verde
        green_upper = np.array([90, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        
        # Dilatar la máscara verde para asegurar que cubrimos todo el pasto
        green_mask = cv2.dilate(green_mask, self.kernel, iterations=2)
        
        # Eliminar áreas de la máscara amarilla que se superponen con la máscara verde
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(green_mask))
        
        # Aplicar operaciones morfológicas para limpiar la máscara
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        
        # Mostrar la máscara verde para depuración
        cv2.imshow("Máscara Verde (Pasto)", green_mask)
        
        # Si estamos siguiendo una ruta y la acción es ir recto, solo detectar en la parte superior
        if self.route and self.turn_action == 'straight':
            # Crear una máscara para mantener solo la parte superior (adelante)
            # Dividir la imagen en tres partes verticales y mantener solo la central
            h_mask, w_mask = mask.shape
            left_limit = int(w_mask * 0.3)
            right_limit = int(w_mask * 0.7)
            
            # Crear una máscara de ceros y copiar solo la región central
            filtered_mask = np.zeros_like(mask)
            filtered_mask[:, left_limit:right_limit] = mask[:, left_limit:right_limit]
            mask = filtered_mask
            
        cv2.imshow("Máscara", mask)

        # Muestreo de filas de máscara para obtener puntos de carril
        ys = np.linspace(0, ch-1, num=self.sample_count, dtype=int)
        raw_pts = []
        for y in ys:
            xs = np.where(mask[y] > 0)[0]
            if xs.size:
                raw_pts.append((int(xs.mean()), y))

        # Planificación de rutas a través de polinomios o líneas
        path_points = []
        if len(raw_pts) >= self.min_samples:
            pts = np.array(raw_pts)
            
            # Mejorar la detección en curvas hacia la izquierda
            # Si estamos en una curva hacia la izquierda, dar más peso a los puntos de la izquierda
            if self.turn_action == 'left' or (self.prev_omega > 0.3 and not self.turn_action):
                # Calcular el centro horizontal de los puntos detectados
                mean_x = np.mean(pts[:, 0])
                # Dar más peso a los puntos a la izquierda del centro
                weights = np.ones(len(pts))
                for i, (x, _) in enumerate(pts):
                    if x < mean_x:  # Si el punto está a la izquierda
                        weights[i] = 2.0  # Dar más peso
                
                # Usar ajuste polinómico ponderado para curvas a la izquierda
                coeffs = np.polyfit(pts[:, 1], pts[:, 0], 2, w=weights)
            else:
                # Ajuste polinómico normal para otros casos
                coeffs = np.polyfit(pts[:, 1], pts[:, 0], 2)
                
            y_vals = np.linspace(0, ch-1, num=self.sample_count)
            x_vals = np.polyval(coeffs, y_vals)
            path_points = [(int(x), int(y)) for x, y in zip(x_vals, y_vals)]
        elif len(raw_pts) >= 2:
            (x0, y0), (x1, y1) = raw_pts[0], raw_pts[-1]
            for t in np.linspace(0, 1, self.sample_count):
                path_points.append((int(x0*(1-t)+x1*t), int(y0*(1-t)+y1*t)))

        # Visualización de la detección
        # calcular desplazamiento horizontal (positivo = derecha) para conducción en el carril derecho
        offset_x = int(cw * self.offset_ratio)
        for x, y in raw_pts:
            cv2.circle(vis, (x + offset_x, y), 3, (255, 0, 255), -1)
        for i in range(1, len(path_points)):
            p1 = (path_points[i-1][0] + offset_x, path_points[i-1][1])
            p2 = (path_points[i][0] + offset_x, path_points[i][1])
            cv2.line(vis, p1, p2, (0, 255, 0), 2)

        # Objetivo de anticipación
        target = None
        if path_points:
            idx = min(len(path_points)-1, int(len(path_points)*self.lookahead_dist))
            target = path_points[idx]
            cv2.circle(vis, (target[0] + offset_x, target[1]), 6, (0, 0, 255), -1)

        # Calcular omega de control con desplazamiento en el carril derecho
        omega = self.prev_omega
        if target is not None:
            # desplazar el objetivo a la derecha por offset_x antes del control
            tx_shifted = target[0] + offset_x
            # centro de la imagen
            center_x = cw // 2
            # error relativo al centro, llevando al bot a seguir la ruta desplazada
            error = (tx_shifted - center_x) / float(cw)
            
            # Aumentar la ganancia para giros a la izquierda para mejorar la respuesta
            kp_adjusted = self.Kp
            if error < 0:  # Error negativo significa que necesitamos girar a la izquierda
                # Aumentar la ganancia proporcional para giros a la izquierda
                kp_adjusted = self.Kp * 1.5
                
                # Si estamos en una curva pronunciada a la izquierda, aumentar aún más
                if error < -0.3:
                    kp_adjusted = self.Kp * 2.0
            
            omega_cmd = -kp_adjusted * error
            omega_clip = float(np.clip(omega_cmd, -self.max_omega, self.max_omega))
            
            # Reducir el factor de suavizado para giros a la izquierda para una respuesta más rápida
            alpha_adjusted = self.alpha
            if error < 0:  # Giro a la izquierda
                alpha_adjusted = min(0.6, self.alpha * 2)  # Mayor valor = respuesta más rápida
                
            omega = alpha_adjusted * omega_clip + (1 - alpha_adjusted) * self.prev_omega
        self.prev_omega = omega

        return self.speed, omega, vis

    def run(self):
        obs = self.safe_reset()
        self.env.render()
        try:
            while True:
                # Seguimiento de carriles y obtener comando
                speed, omega, vis = self.process_frame(obs)
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
                    print("Episodio reiniciado")

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
