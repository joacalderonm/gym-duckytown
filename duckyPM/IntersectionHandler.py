import numpy as np

class IntersectionHandler:
    def __init__(self, sim):
        self.sim = sim
        self.red_count_history = []
        self.red_history_size = 10
        self.red_peak_detected = False
        self.frames_after_peak = 0
        self.turn_start_frame = 0
        self.intersection_cooldown = 0

    def handle_intersection(self, red_count):
        """Maneja el estado de la intersección basado en el conteo de la línea roja."""

        speed = self.sim.speed  # Valor predeterminado si no hay intersección
        omega = self.sim.prev_omega  # Valor predeterminado para omega

        # Asegúrate de que speed no sea menor a un valor mínimo (por ejemplo, 0.1)
        if speed < 0.1:
            speed = 0.1

        # Asegúrate de que omega no sea demasiado pequeño
        if abs(omega) < 0.05:
            omega = 0.05

        # Disminuir contador de enfriamiento si está activo
        if self.intersection_cooldown > 0:
            self.intersection_cooldown -= 1
            if self.intersection_cooldown == 0:
                self.sim.logger.info("Enfriamiento de intersección terminado, listo para la próxima intersección")
        
        if self.sim.state == 'FOLLOW':
            # Detectar línea roja y prepararse para la intersección (solo si no está en enfriamiento)
            if red_count > self.sim.red_threshold and self.intersection_cooldown == 0:
                self.sim.logger.info("Línea roja detectada, preparándose para la intersección")
                self.sim.state = 'APPROACHING'
                speed = self.sim.intersection_speed * 1.5  # Aumentar velocidad al acercarse a la intersección
                omega = self.sim.prev_omega * 0.7  # Reducir dirección para acercarse más recto
                
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
            if len(self.red_count_history) >= 3:
                if not self.red_peak_detected and self.red_count_history[-1] < self.red_count_history[-2]:
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
            return speed, self.sim.prev_omega * 0.7  # Reducir dirección para acercarse más recto
        
        elif self.sim.state == 'TURNING':
            # Ejecutar el giro después de cruzar la línea roja
            speed = self.sim.intersection_speed
            
            # Aplicar dirección apropiada según la dirección del giro
            if self.sim.turn_action == 'left':
                turn_progress = min(1.0, self.turn_start_frame / 3.0)  # Aumento gradual para giros a la izquierda
                omega = self.sim.turn_omega * 1.3 * turn_progress  # 30% más de velocidad angular
            elif self.sim.turn_action == 'right':
                turn_progress = min(1.0, self.turn_start_frame / 5.0)
                omega = -self.sim.turn_omega * turn_progress
            else:  # recto
                omega = 0.1  # Ligera corrección para mantener la trayectoria recta
            
            # Incrementar contador de cuadros de giro
            self.turn_start_frame += 1
            
            # Salir del estado de giro después de suficientes cuadros
            turn_duration_adjusted = self.sim.turn_duration
            if self.sim.turn_action == 'left':
                turn_duration_adjusted = int(self.sim.turn_duration * 1.5)  # 50% más de tiempo para giros a la izquierda
            
            if self.turn_start_frame >= turn_duration_adjusted:
                self.sim.logger.info(f"Giro {self.sim.turn_action} completado, regresando al seguimiento de carriles")
                self.sim.state = 'FOLLOW'
                self.sim.turn_action = None
                self.sim.intersection_cooldown = self.sim.cooldown_duration
                self.sim.logger.info(f"Estableciendo enfriamiento de intersección por {self.sim.cooldown_duration} cuadros")
            
            return speed, omega
        return 0, 0  # Si no es ningún estado válido, retornamos valores neutros
