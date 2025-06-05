import cv2
import numpy as np


class VisionProcessor:
    def __init__(self, sim):
        self.sim = sim

    def _preprocess_image(self, frame):
        """
        Convierte imagen RGB a HSV, recorta el cielo y aplica desenfoque.
        Devuelve: imagen HSV, lienzo visual y shape de imagen.
        """
        # 1. Convertir a BGR (por OpenCV)
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 2. Recortar parte superior
        h, w, _ = bgr.shape
        crop_top = h // 3
        crop = bgr[crop_top:, :]

        # 3. Copia visual
        vis = crop.copy()

        # 4. Desenfoque para reducir ruido
        blur = cv2.GaussianBlur(crop, (5, 5), 0)

        # 5. Convertir a HSV para detección de colores
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        return hsv, vis, crop.shape[:2]

    def _detect_red_intersection(self, hsv):
        """
        Detecta intersecciones a través de franjas rojas.
        Devuelve: máscara roja procesada y conteo de píxeles rojos.
        """
        sim = self.sim

        # Detección de franjas rojas
        red_mask1 = cv2.inRange(hsv, sim.red_lower1, sim.red_upper1)
        red_mask2 = cv2.inRange(hsv, sim.red_lower2, sim.red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        # Limpiar la máscara roja con operaciones morfológicas
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, sim.kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, sim.kernel)

        # Contar píxeles rojos
        red_count = cv2.countNonZero(red_mask)
        cv2.imshow("Máscara Roja", red_mask)

        # Actualizar historial de conteo rojo para detección de picos
        sim.red_count_history.append(red_count)
        if len(sim.red_count_history) > sim.red_history_size:
            sim.red_count_history.pop(0)

        # Control de cooldown de intersección
        if sim.intersection_cooldown > 0:
            sim.intersection_cooldown -= 1
            if sim.intersection_cooldown == 0:
                sim.logger.info("Enfriamiento de intersección terminado, listo para la próxima intersección")

        # Máquina de estados mejorada para manejo de intersecciones
        if sim.state == 'FOLLOW' and red_count > sim.red_threshold and sim.intersection_cooldown == 0:
            sim.logger.info("Línea roja detectada, preparándose para la intersección")
            sim.state = 'APPROACHING'
            sim.turn_action = sim.route[sim.route_index % len(sim.route)] if sim.route else np.random.choice(['left', 'right', 'straight'])
            sim.route_index += 1
            sim.logger.info(f"Siguiendo ruta: giro {sim.turn_action} (índice {sim.route_index-1})")
            sim.red_peak_detected = False
            sim.frames_after_peak = 0
        
        elif sim.state == 'APPROACHING':
            # Reducir velocidad al acercarse a la intersección
            speed = sim.intersection_speed * 1.5  # Un poco más rápido que la velocidad de cruce
            # Verificar si hemos pasado el pico de rojo
            if len(sim.red_count_history) >= 3 and not self.peak_red_detected:
                if sim.red_count_history[-1] < sim.red_count_history[-2]:
                    # Hemos pasado el pico de rojo
                    sim.red_peak_detected = True
                    sim.logger.info("Pico rojo detectado, esperando para cruzar la línea")
            
            # Después de detectar el pico, contar cuadros hasta que estemos más allá de la línea
            if sim.red_peak_detected:
                sim.frames_after_peak += 1
                
                # Una vez que hemos esperado suficientes cuadros después del pico, comenzar a girar
                if sim.frames_after_peak >= sim.frames_to_wait:
                    sim.logger.info(f"Comenzando giro {sim.turn_action}")
                    sim.state = 'TURNING'
                    self.turn_start_frame = 0
            
            # Continuar con el seguimiento de carriles mientras se acerca
            return speed, sim.prev_omega * 0.7  # Reducir dirección para acercarse más recto
        
        elif sim.state == 'TURNING':
            # Ejecutar el giro después de cruzar la línea roja
            speed = sim.intersection_speed

            # Aplicar dirección apropiada según la dirección del giro
            if sim.turn_action == 'left':
                omega = sim.turn_omega * 1.3 * min(1.0, self.turn_start_frame / 3.0)  # Aumento más agresivo para giros a la izquierda
            elif sim.turn_action == 'right':
                omega = -sim.turn_omega * min(1.0, self.turn_start_frame / 5.0)
            else:  # recto
                omega = 0.1  # Ligera corrección hacia la izquierda para mantener la trayectoria recta

            return speed, omega 
        
        return None
    
    def _detect_yellow_lane(self, hsv, shape):
        """
        Detecta el carril amarillo eliminando zonas verdes (pasto).
        Devuelve: máscara final del carril y lista de puntos detectados.
        """
        sim = self.sim
        ch, cw = shape

        # Detección de línea amarilla
        mask = cv2.inRange(hsv, sim.lower_bound, sim.upper_bound)

        # Filtrar áreas que probablemente sean pasto (verde)
        green_lower = np.array([35, 50, 50])
        green_upper = np.array([90, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)

        # Dilatar la máscara verde para asegurar que cubrimos todo el pasto
        green_mask = cv2.dilate(green_mask, sim.kernel, iterations=2)

        # Eliminar áreas de la máscara amarilla que se superponen con la máscara verde
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(green_mask))

        # Aplicar operaciones morfológicas para limpiar la máscara
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, sim.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, sim.kernel)
        
        # Mostrar la máscara verde para depuración
        cv2.imshow("Máscara Verde (Pasto)", green_mask)

        # Si estamos siguiendo una ruta y la acción es ir recto, solo detectar en la parte superior
        if sim.route and sim.turn_action == 'straight':
            # Crear una máscara para mantener solo la parte superior (adelante)
            h_mask, w_mask = mask.shape
            left_limit = int(w_mask * 0.3)
            right_limit = int(w_mask * 0.7)

            # Crear una máscara de ceros y copiar solo la región central
            filtered_mask = np.zeros_like(mask)
            filtered_mask[:, left_limit:right_limit] = mask[:, left_limit:right_limit]
            mask = filtered_mask

        cv2.imshow("Máscara", mask)
        # Muestreo de filas de máscara para obtener puntos de carril

        ys = np.linspace(0, ch - 1, num=sim.sample_count, dtype=int)
        raw_pts = []
        for y in ys:
            xs = np.where(mask[y] > 0)[0]
            if xs.size:
                raw_pts.append((int(xs.mean()), y))
                
            


    

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
            red_mask1 = cv2.inRange(hsv, self.sim.red_lower1, self.sim.red_upper1)
            red_mask2 = cv2.inRange(hsv, self.sim.red_lower2, self.sim.red_upper2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, self.sim.kernel)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, self.sim.kernel)
            red_count = cv2.countNonZero(red_mask)
            cv2.imshow("Máscara Roja", red_mask)
            
            # Actualizar historial de conteo rojo para detección de picos
            self.sim.red_count_history.append(red_count)
            if len(self.sim.red_count_history) > self.sim.red_history_size:
                self.sim.red_count_history.pop(0)
            
            # Disminuir contador de enfriamiento si está activo
            if self.sim.intersection_cooldown > 0:
                self.sim.intersection_cooldown -= 1
                if self.sim.intersection_cooldown == 0:
                    self.sim.logger.info("Enfriamiento de intersección terminado, listo para la próxima intersección")
            
            # Máquina de estados mejorada para manejo de intersecciones
            if self.sim.state == 'FOLLOW':
                # Detectar línea roja y prepararse para la intersección (solo si no está en enfriamiento)
                if red_count > self.sim.red_threshold and self.sim.intersection_cooldown == 0:
                    self.sim.logger.info("Línea roja detectada, preparándose para la intersección")
                    self.sim.state = 'APPROACHING'
                    
                    # Obtener el siguiente giro de la ruta si está disponible, de lo contrario usar elección aleatoria
                    if self.sim.route:
                        self.sim.turn_action = self.sim.route[self.sim.route_index % len(self.sim.route)]
                        self.sim.route_index += 1
                        self.sim.logger.info(f"Siguiendo ruta: giro {self.sim.turn_action} (índice {self.sim.route_index-1})")
                    else:
                        self.sim.turn_action = np.random.choice(['straight', 'right', 'left'])
                        self.sim.logger.info(f"Giro aleatorio: {self.sim.turn_action}")
                    
                    self.red_peak_detected = False
                    self.sim.frames_after_peak = 0
                    
            elif self.sim.state == 'APPROACHING':
                # Reducir velocidad al acercarse a la intersección
                speed = self.sim.intersection_speed * 1.5  # Un poco más rápido que la velocidad de cruce
                
                # Verificar si hemos pasado el pico de rojo (lo que significa que estamos cruzando la línea)
                if len(self.sim.red_count_history) >= 3:
                    if not self.red_peak_detected and self.sim.red_count_history[-1] < self.sim.red_count_history[-2]:
                        # Hemos pasado el pico de rojo
                        self.red_peak_detected = True
                        self.sim.logger.info("Pico rojo detectado, esperando para cruzar la línea")
                    
                # Después de detectar el pico, contar cuadros hasta que estemos más allá de la línea
                if self.red_peak_detected:
                    self.sim.frames_after_peak += 1
                    
                    # Una vez que hemos esperado suficientes cuadros después del pico, comenzar a girar
                    if self.sim.frames_after_peak >= self.sim.frames_to_wait:
                        self.sim.logger.info(f"Comenzando giro {self.sim.turn_action}")
                        self.sim.state = 'TURNING'
                        self.turn_start_frame = 0
                
                # Continuar con el seguimiento de carriles mientras se acerca
                return speed, self.sim.prev_omega * 0.7, vis  # Reducir dirección para acercarse más recto
                
            elif self.sim.state == 'TURNING':
                # Ejecutar el giro después de cruzar la línea roja
                speed = self.sim.intersection_speed
                
                # Aplicar dirección apropiada según la dirección del giro
                if self.sim.turn_action == 'left':
                    # Aumento gradual de la tasa de giro para giros a la izquierda
                    # Usar una rampa más agresiva para giros a la izquierda
                    turn_progress = min(1.0, self.turn_start_frame / 3.0)  # Más rápido que 5.0
                    # Aumentar la velocidad angular para giros a la izquierda
                    omega = self.sim.turn_omega * 1.3 * turn_progress  # 30% más de velocidad angular
                elif self.sim.turn_action == 'right':
                    # Aumento gradual de la tasa de giro
                    turn_progress = min(1.0, self.turn_start_frame / 5.0)
                    omega = -self.sim.turn_omega * turn_progress
                else:  # recto
                    # Aplicar una pequeña corrección para mantener el robot centrado
                    # Esto ayuda a contrarrestar la tendencia a desviarse hacia la derecha
                    omega = 0.1  # Ligera corrección hacia la izquierda para mantener la trayectoria recta
                
                # Incrementar contador de cuadros de giro
                self.turn_start_frame += 1
                
                # Salir del estado de giro después de suficientes cuadros
                # Para giros a la izquierda, aumentar la duración del giro
                turn_duration_adjusted = self.sim.turn_duration
                if self.sim.turn_action == 'left':
                    turn_duration_adjusted = int(self.sim.turn_duration * 1.5)  # 50% más de tiempo para giros a la izquierda
                    
                if self.turn_start_frame >= turn_duration_adjusted:
                    self.sim.logger.info(f"Giro {self.sim.turn_action} completado, regresando al seguimiento de carriles")
                    self.sim.state = 'FOLLOW'
                    self.sim.turn_action = None
                    # Establecer enfriamiento para evitar la detección inmediata de la intersección
                    self.sim.intersection_cooldown = self.sim.cooldown_duration
                    self.sim.logger.info(f"Estableciendo enfriamiento de intersección por {self.sim.cooldown_duration} cuadros")
                
                return speed, omega, vis

            # Detección de línea amarilla con filtrado mejorado para evitar el pasto
            mask = cv2.inRange(hsv, self.sim.lower_bound, self.sim.upper_bound)
            
            # Filtrar áreas que probablemente sean pasto (verde)
            # Crear una máscara para el color verde (pasto)
            green_lower = np.array([35, 50, 50])  # Verde
            green_upper = np.array([90, 255, 255])
            green_mask = cv2.inRange(hsv, green_lower, green_upper)
            
            # Dilatar la máscara verde para asegurar que cubrimos todo el pasto
            green_mask = cv2.dilate(green_mask, self.sim.kernel, iterations=2)
            
            # Eliminar áreas de la máscara amarilla que se superponen con la máscara verde
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(green_mask))
            
            # Aplicar operaciones morfológicas para limpiar la máscara
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.sim.kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.sim.kernel)
            
            # Mostrar la máscara verde para depuración
            cv2.imshow("Máscara Verde (Pasto)", green_mask)
            
            # Si estamos siguiendo una ruta y la acción es ir recto, solo detectar en la parte superior
            if self.sim.route and self.sim.turn_action == 'straight':
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
            ys = np.linspace(0, ch-1, num=self.sim.sample_count, dtype=int)
            raw_pts = []
            for y in ys:
                xs = np.where(mask[y] > 0)[0]
                if xs.size:
                    raw_pts.append((int(xs.mean()), y))

            # Planificación de rutas a través de polinomios o líneas
            path_points = []
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
                    coeffs = np.polyfit(pts[:, 1], pts[:, 0], 2, w=weights)
                else:
                    # Ajuste polinómico normal para otros casos
                    coeffs = np.polyfit(pts[:, 1], pts[:, 0], 2)
                    
                y_vals = np.linspace(0, ch-1, num=self.sim.sample_count)
                x_vals = np.polyval(coeffs, y_vals)
                path_points = [(int(x), int(y)) for x, y in zip(x_vals, y_vals)]
            elif len(raw_pts) >= 2:
                (x0, y0), (x1, y1) = raw_pts[0], raw_pts[-1]
                for t in np.linspace(0, 1, self.sim.sample_count):
                    path_points.append((int(x0*(1-t)+x1*t), int(y0*(1-t)+y1*t)))

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
                cv2.circle(vis, (target[0] + offset_x, target[1]), 6, (0, 0, 255), -1)

            # Calcular omega de control con desplazamiento en el carril derecho
            omega = self.sim.prev_omega
            if target is not None:
                # desplazar el objetivo a la derecha por offset_x antes del control
                tx_shifted = target[0] + offset_x
                # centro de la imagen
                center_x = cw // 2
                # error relativo al centro, llevando al bot a seguir la ruta desplazada
                error = (tx_shifted - center_x) / float(cw)
                
                # Aumentar la ganancia para giros a la izquierda para mejorar la respuesta
                kp_adjusted = self.sim.Kp
                if error < 0:  # Error negativo significa que necesitamos girar a la izquierda
                    # Aumentar la ganancia proporcional para giros a la izquierda
                    kp_adjusted = self.sim.Kp * 1.5
                    
                    # Si estamos en una curva pronunciada a la izquierda, aumentar aún más
                    if error < -0.3:
                        kp_adjusted = self.sim.Kp * 2.0
                
                omega_cmd = -kp_adjusted * error
                omega_clip = float(np.clip(omega_cmd, -self.sim.max_omega, self.sim.max_omega))
                
                # Reducir el factor de suavizado para giros a la izquierda para una respuesta más rápida
                alpha_adjusted = self.sim.alpha
                if error < 0:  # Giro a la izquierda
                    alpha_adjusted = min(0.6, self.sim.alpha * 2)  # Mayor valor = respuesta más rápida
                    
                omega = alpha_adjusted * omega_clip + (1 - alpha_adjusted) * self.sim.prev_omega
            self.sim.prev_omega = omega

            return self.sim.speed, omega, vis
