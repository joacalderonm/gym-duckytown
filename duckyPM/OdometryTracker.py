import numpy as np

class Odometria:
    def __init__(self, sim):
        self.sim = sim
        self.x = 0.0  # Posición en X
        self.y = 0.0  # Posición en Y
        self.theta = 0.0  # Orientación del robot
        self.baseline = sim.baseline  # Distancia entre ruedas (en metros)
        self.dt = 1.0 / sim.frame_rate if sim.frame_rate else 0.1  # Tiempo por frame

    def update_odometry(self, speed, omega):
        """
        Actualiza la odometría (posición y orientación) usando el desplazamiento calculado.
        """
        v_l = speed - omega * (self.baseline / 2)  # Velocidad de la rueda izquierda
        v_r = speed + omega * (self.baseline / 2)  # Velocidad de la rueda derecha

        # Desplazamientos de las ruedas
        d_l = v_l * self.dt  # Desplazamiento de la rueda izquierda
        d_r = v_r * self.dt  # Desplazamiento de la rueda derecha
        d_center = (d_r + d_l) / 2.0  # Desplazamiento total (promedio de las ruedas)
        d_theta = (d_r - d_l) / self.baseline  # Cambio de orientación (basado en la diferencia de las ruedas)

        # Actualización de la posición y orientación
        self.theta += d_theta  # Actualización del ángulo
        self.x += d_center * np.cos(self.theta)  # Desplazamiento en X
        self.y += d_center * np.sin(self.theta)  # Desplazamiento en Y

    def get_position(self):
        """
        Devuelve la posición actual del robot (x, y) y su orientación (theta).
        """
        return self.x, self.y, self.theta

    def get_target_point(self, raw_pts):
        """
        Calcula el punto objetivo basándose en los puntos del carril.
        """
        if not raw_pts:
            return None

        # Elegir el punto objetivo basándose en la distancia anticipada
        idx = min(len(raw_pts) - 1, int(len(raw_pts) * self.sim.lookahead_dist))

        target = raw_pts[idx]  # Este es el punto objetivo
        return target
