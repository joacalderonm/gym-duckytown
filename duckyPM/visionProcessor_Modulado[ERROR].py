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

            self.turn_start_frame += 1

            # Salir del estado de giro después de suficientes cuadros
            turn_duration_adjusted = sim.turn_duration
            if sim.turn_action == 'left':
                turn_duration_adjusted = int(sim.turn_duration * 1.5)

            if self.turn_start_frame >= turn_duration_adjusted:
                sim.logger.info(f"Giro {sim.turn_action} completado, regresando al seguimiento de carriles")
                sim.state = 'FOLLOW'
                sim.turn_action = None
                # Establecer enfriamiento para evitar la detección inmediata de la intersección
                sim.intersection_cooldown = sim.cooldown_duration
                sim.logger.info(f"Estableciendo enfriamiento de intersección por {sim.cooldown_duration} cuadros")

            # Devolver velocidad y omega
            self.sim.prev_omega = omega
            speed = sim.speed
            omega = omega * 0.7
            
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
        
        # Si estamos siguiendo una ruta y la acción es ir recto, solo detectar en la parte superior
        if sim.route and sim.turn_action == 'straight':
            # Crear una máscara para mantener solo la parte superior (adelante)
            left_limit = int(cw * 0.3)
            right_limit = int(ch * 0.7)

            # Crear una máscara de ceros y copiar solo la región central
            filtered_mask = np.zeros_like(mask)
            filtered_mask[:, left_limit:right_limit] = mask[:, left_limit:right_limit]
            mask = filtered_mask

        ys = np.linspace(0, ch - 1, num=sim.sample_count, dtype=int)
        raw_pts = []
        for y in ys:
            xs = np.where(mask[y] > 0)[0]
            if xs.size:
                raw_pts.append((int(xs.mean()), y))

        cv2.imshow("Máscara amarilla", mask)

        return mask, raw_pts
    
    def _compute_control(self, target, shape):
        """
        Calcula el control omega basado en el objetivo detectado.
        Devuelve: omega ajustado y visualización.
        """
        sim = self.sim
        ch, cw = shape
        offset_x = int(cw * sim.offset_ratio)
        tx_shifted = target[0] + offset_x
        center_x = cw // 2
        error = (tx_shifted - center_x) / float(cw)

        # Ajuste de ganancia
        kp_adjusted = sim.Kp * (2.0 if error < -0.3 else 1.5 if error < 0 else 1.0)
        omega_cmd = -kp_adjusted * error
        omega_clip = float(np.clip(omega_cmd, -sim.max_omega, sim.max_omega))

        # Ajuste de suavizado
        alpha_adjusted = min(0.6, sim.alpha * 2) if error < 0 else sim.alpha
        omega = alpha_adjusted * omega_clip + (1 - alpha_adjusted) * sim.prev_omega
        sim.prev_omega = omega

        return omega            
    
    def _get_target_point(self, raw_pts):
        """
        Obtiene el punto objetivo basado en los puntos del carril.
        El objetivo es anticipar hacia donde debe ir el robot.
        """
        sim = self.sim

        if not raw_pts:
            return None

        # Elegir el punto objetivo basándose en la distancia anticipada
        idx = min(len(raw_pts) - 1, int(len(raw_pts) * sim.lookahead_dist))

        target = raw_pts[idx]
        return target


    def process_frame(self, frame):
        """
        Procesa un frame: preprocesamiento, detección de intersección y control.
        """
        # 1. Preprocesamiento de la imagen (convierte, recorta, desenfoque)
        hsv, vis, shape = self._preprocess_image(frame)

        # 2. Detección de intersección (líneas rojas)
        inter_result = self._detect_red_intersection(hsv)
        if inter_result:
            speed, omega = inter_result
            return speed, omega, vis

        # 3. Detección del carril amarillo (máscara amarilla, eliminación de pasto)
        lane_mask, raw_pts = self._detect_yellow_lane(hsv, shape)

        # 4. Planificación de trayectoria o control de dirección
        if len(raw_pts) > 0:
            target = self._get_target_point(raw_pts)  # Obtiene el punto objetivo para el bot
            omega = self._compute_control(target, shape)  # Calcula omega basado en el objetivo
        else:
            omega = self.sim.prev_omega  # Si no detecta carril, mantiene el valor anterior de omega

        return self.sim.speed, omega, vis
