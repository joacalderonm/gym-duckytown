import cv2
import numpy as np

class VisionProcessor:
    def __init__(self, sim):
        self.sim = sim    
        self.logger = self.sim.logger

    def process_frame(self, frame):
        """Preprocesa el frame para detección de carriles y de intersecciones."""
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        h, w, _ = bgr.shape
        crop_top = h // 3
        crop = bgr[crop_top:, :]
        ch, cw = crop.shape[:2]

        # lienzo de visualización para este cuadro
        vis = crop.copy()

        # Aplicar desenfoque gaussiano para reducir el ruido
        blur = cv2.GaussianBlur(crop, (5, 5), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        speed, omega = self.detect_intersections(hsv)
        
        # Detección de carriles amarillos (máscara amarilla)
        mask = cv2.inRange(hsv, self.sim.lower_bound, self.sim.upper_bound)
        
        # Filtrar áreas que probablemente sean pasto (verde)
        green_lower = np.array([35, 50, 50])  # Ajuste el valor mínimo para el verde
        green_upper = np.array([90, 255, 255])  # Ajuste el valor máximo para el verde
        green_mask = cv2.inRange(hsv, green_lower, green_upper)

        # Dilatar la máscara verde para asegurar que cubrimos todo el pasto
        green_mask = cv2.dilate(green_mask, self.sim.kernel, iterations=2)

        # Eliminar áreas de la máscara amarilla que se superponen con el pasto
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(green_mask))

        # Mostrar la máscara verde para depuración
        cv2.imshow("Máscara Verde (Pasto)", green_mask)

        # Asegurarse de que los valores de speed y omega no sean None ni muy pequeños
        speed = self.sim.speed  # Valor predeterminado
        omega = self.sim.prev_omega  # Valor predeterminado

        if self.sim.route and self.sim.turn_action == 'straight':
            # Crear una máscara para mantener solo la parte superior (adelante)
            # Dividir la imagen en tres partes vertcales y mantener solo la central
            h_mask, w_mask = mask.shape
            left_limit = int(w_mask * 0.3)  # 30% del ancho a la izquierda
            right_limit = int(w_mask * 0.7)  # 30% del ancho a la derecha

            # Crear una máscara de ceros y copiar solo la región central
            filtered_mask = np.zeros_like(mask)
            filtered_mask[:, left_limit:right_limit] = mask[:, left_limit:right_limit]
            mask = filtered_mask  # Actualizar la máscara con la región central
        
        cv2.imshow("Máscara", mask)

        # Muestreo de filas de la máscara para obtener puntos de carril
        ys = np.linspace(0, ch-1, num=self.sim.sample_count, dtype=int)  # Muestreo de filas
        raw_pts = []  # Lista para almacenar los puntos de los carriles
        for y in ys:
            # Buscar las posiciones de los píxeles amarillos en la fila y
            xs = np.where(mask[y] > 0)[0]  # Obtener los índices de los píxeles amarillos
            if xs.size:  # Si encontramos píxeles amarillos
                raw_pts.append((int(xs.mean()), y))  # Añadir el punto medio de los carriles a la lista

        # Planificar ruta a través de polinomias o líneas
        path_points = []

        # Verifica que haya suficientes puntos de carril para realizar el ajuste polinómico
        if len(raw_pts) >= self.sim.min_samples:
            pts = np.array(raw_pts)

            # Mejorar la detección en curvas hacia la izquierda
            # Si estamos en una curva hacia la izquierda, dar más peso a los puntos de la izquierda
            if self.sim.turn_action == 'left' or (self.sim.prev_omega > 0.3 and not self.sim.turn_action):
                # Calcular el centro horizontal de los puntos detectados
                mean_x = np.mean(pts[:, 0])
                # Dar más peso a los puntos a la izquierda del centro
                weights = np.ones(len(pts))
                for i, (x, _) in enumerate(pts):
                    if x < mean_x:  # Si el punto está a la izquierda
                        weights[i] = 2.0  # Dar más peso

                # Usar ajuste polinómico ponderado para curvas a la izquierda
                try:
                    coeffs = np.polyfit(pts[:, 1], pts[:, 0], 2, w=weights)
                except np.linalg.LinAlgError:
                    self.logger.error("Error en el ajuste polinómico ponderado. Usando ajuste lineal.")
                    coeffs = np.polyfit(pts[:, 1], pts[:, 0], 1)  # Ajuste lineal como respaldo
            else:
                # Ajuste polinómico normal para otros casos
                try:
                    coeffs = np.polyfit(pts[:, 1], pts[:, 0], 2)
                except np.linalg.LinAlgError:
                    self.logger.error("Error en el ajuste polinómico. Usando ajuste lineal.")
                    coeffs = np.polyfit(pts[:, 1], pts[:, 0], 1)  # Ajuste lineal como respaldo

        elif len(raw_pts) >= 2:
            (x0, y0), (x1, y1) = raw_pts[0], raw_pts[-1]
            for t in np.linspace(0, 1, num=self.sim.sample_count):
                x = int((1 - t) * x0 + t * x1)
                y = int((1 - t) * y0 + t * y1)
                path_points.append((x, y))

        # Visualización de la detección
        # calcular desplazamiento horizontal (positivo = derecha) para conducción en el carril derecho
        offset_x = int(cw * self.sim.offset_ratio)
        for x, y in raw_pts:
            cv2.circle(vis, (x + offset_x, y), 3, (255, 0, 255), -1)
        for i in range(1, len(path_points)):
            p1 = (path_points[i-1][0] + offset_x, path_points[i-1][1])
            p2 = (path_points[i][0] + offset_x, path_points[i][1])
            cv2.line(vis, p1, p2, (0, 255, 0), 2)
    
        # Objetivo de anticipación
        target = None
        if path_points:
            idx = min(len(path_points)-1, int(len(path_points)*self.sim.lookahead_dist))
            target = path_points[idx]
            cv2.circle(vis, (target[0], target[1]), 6, (0, 0, 255), -1)

        # Cálculo de omega con control proporcional
        omega = self.sim.prev_omega
        if target is not None:
            # desplazar el objetivo a la derecha por offset_x antes del control
            tx_shifted = target[0]
            # centro de la imagen
            center_x = cw // 2
            # error relativo al centro, llevando al bot a seguir la ruta desplazada
            error = (tx_shifted - center_x) / float(cw)

            # Aumentar la ganancia para giros a la izquierda para mejorar la respuesta
            kp_adjusted = self.sim.Kp
            if error < 0:  # Giro a la izquierda
                kp_adjusted = self.sim.Kp * 1.5

                if error < -0.3:
                    kp_adjusted = self.sim.Kp * 2.0

            omega_cmd = -kp_adjusted * error
            omega_clip = float(np.clip(omega_cmd, -self.sim.max_omega, self.sim.max_omega))

            # Reducir el factor de suavizado para giros a la izquierda para una respuesta más rápida
            alpha_adjusted = self.sim.alpha
            if error < 0:  # Giro a la izquierda
                alpha_adjusted = min(0.6, self.sim.alpha * 2)

            omega = alpha_adjusted * omega_clip + (1 - alpha_adjusted) * self.sim.prev_omega
        self.sim.prev_omega = omega

        # Mostrar la imagen con la trayectoria
        cv2.imshow("Trayectoria", vis)

        return self.sim.speed, omega, vis


    def detect_intersections(self, hsv):
        """Detecta intersecciones a través de franjas rojas y maneja el estado del robot."""
        # Detección de franjas rojas para intersecciones
        red_mask1 = cv2.inRange(hsv, self.sim.red_lower1, self.sim.red_upper1)
        red_mask2 = cv2.inRange(hsv, self.sim.red_lower2, self.sim.red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, self.sim.kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, self.sim.kernel)
        red_count = cv2.countNonZero(red_mask)

        # Mostrar la máscara roja para depuración
        cv2.imshow("Máscara Roja", red_mask)

        # Actualizar historial de conteo rojo para detección de picos
        self.sim.red_count_history.append(red_count)
        if len(self.sim.red_count_history) > self.sim.red_history_size:
            self.sim.red_count_history.pop(0)

        # Disminuir contador de enfriamiento si está activo
        if self.sim.intersection_cooldown > 0:
            self.sim.intersection_cooldown -= 1
            if self.sim.intersection_cooldown == 0:
                self.logger.info("Enfriamiento de intersección terminado, listo para la próxima intersección")

        # Máquina de estados mejorada para manejo de intersecciones
        if self.sim.state == 'FOLLOW':
            # Detectar línea roja y prepararse para la intersección (solo si no está en enfriamiento)
            if red_count > self.sim.red_threshold and self.sim.intersection_cooldown == 0:
                self.logger.info("Línea roja detectada, preparándose para la intersección")
                self.sim.state = 'APPROACHING'
            
                # Obtener el siguiente giro de la ruta si está disponible, de lo contrario usar elección aleatoria
                if self.sim.route:
                    self.sim.turn_action = self.sim.route[self.sim.route_index % len(self.sim.route)]
                    self.sim.route_index += 1
                    self.logger.info(f"Siguiendo ruta: giro {self.sim.turn_action} (índice {self.sim.route_index-1})")
                else:
                    self.sim.turn_action = np.random.choice(['straight', 'right', 'left'])
                    self.logger.info(f"Giro aleatorio: {self.sim.turn_action}")
                
                self.sim.red_peak_detected = False
                self.sim.frames_after_peak = 0

        elif self.sim.state == 'APPROACHING':
            # Reducir velocidad al acercarse a la intersección
            speed = self.sim.intersection_speed * 1.5  # Un poco más rápido que la velocidad de cruce

            if len(self.sim.red_count_history) >= 3:
                if not self.sim.red_peak_detected and self.sim.red_count_history[-1] < self.sim.red_count_history[-2]:
                    # Hemos pasado el pico de rojo
                    self.sim.red_peak_detected = True
                    self.logger.info("Pico rojo detectado, esperando para cruzar la línea")

            # Después de detectar el pico, contar cuadros hasta que estemos más allá de la línea
            if self.sim.red_peak_detected:
                self.sim.frames_after_peak += 1
                # Una vez que hemos esperado suficientes cuadros después del pico, comenzar a girar
                if self.sim.frames_after_peak >= self.sim.frames_to_wait:
                    self.logger.info(f"Comenzando giro {self.sim.turn_action}")
                    self.sim.state = 'TURNING'
                    self.sim.turn_start_frame = 0

            # Continuar con el seguimiento de carriles mientras se acerca
            return self.sim.speed, self.sim.prev_omega * 0.7, self.vis

        elif self.sim.state == 'TURNING':
            # Ejecutar el giro después de cruzar la línea roja
            speed = self.sim.intersection_speed

            # Control de la dirección según el giro
            if self.sim.turn_action == 'left':
                # Aumento gradual de la tasa de giro para giros a la izquierda
                turn_progress = min(1.0, self.sim.turn_start_frame / 3.0)  # Más rápido que 5.0
                omega = self.sim.turn_omega * 1.3 * turn_progress  # Aumentar la velocidad angular para giros a la izquierda
            elif self.sim.turn_action == 'right':
                # Aumento gradual de la tasa de giro
                turn_progress = min(1.0, self.sim.turn_start_frame / 5.0)
                omega = -self.sim.turn_omega * turn_progress
            else:  # recto
                omega = 0.1  # Ligera corrección hacia la izquierda para mantener la trayectoria recta

            # Incrementar contador de cuadros de giro
            self.sim.turn_start_frame += 1
            # Salir del estado de giro después de suficientes cuadros
            turn_duration_adjusted = self.sim.turn_duration
            if self.sim.turn_action == 'left':
                turn_duration_adjusted = int(self.sim.turn_duration * 1.5)  # 50% más de tiempo para giros a la izquierda

            if self.sim.turn_start_frame >= turn_duration_adjusted:
                self.logger.info(f"Giro {self.sim.turn_action} completado, regresando al seguimiento de carriles")
                self.sim.state = 'FOLLOW'
                self.sim.turn_action = None
                self.sim.intersection_cooldown = self.sim.cooldown_duration
                self.logger.info(f"Estableciendo enfriamiento de intersección por {self.sim.cooldown_duration} cuadros")

            return speed, omega, self.vis

        return self.sim.speed, self.sim.prev_omega
