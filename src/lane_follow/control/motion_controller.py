import numpy as np

class MotionController:
    def __init__(self, baseline=0.08, dt=1.0/30.0):
        self.baseline = baseline  # distancia entre ruedas (m)
        self.dt = dt  # intervalo de tiempo (s)
        
        # Estado de odometría
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_theta = 0.0
        
        # Historial de trayectoria
        self.traj = []
        
        # Control de movimiento
        self.prev_omega = 0.0

    def update_odometry(self, speed, omega):
        # Integración de odometría (tracción diferencial)
        v_l = speed - omega * (self.baseline / 2)
        v_r = speed + omega * (self.baseline / 2)
        d_l = v_l * self.dt
        d_r = v_r * self.dt
        d_center = (d_r + d_l) / 2.0
        d_theta = (d_r - d_l) / self.baseline
        
        # Actualizar pose
        self.odom_theta += d_theta
        self.odom_x += d_center * np.cos(self.odom_theta)
        self.odom_y += d_center * np.sin(self.odom_theta)
        
        # Actualizar trayectoria
        self.traj.append((self.odom_x, self.odom_y))
        
        return self.odom_x, self.odom_y, self.odom_theta

    def compute_control(self, path_points, lookahead_dist, kp, alpha, max_omega, offset_x, cw):
        omega = self.prev_omega
        
        if path_points:
            # Calcular punto objetivo
            idx = min(len(path_points)-1, int(len(path_points)*lookahead_dist))
            target = path_points[idx]
            
            # Calcular error
            tx_shifted = target[0] + offset_x
            center_x = cw // 2
            error = (tx_shifted - center_x) / float(cw)
            
            # Ajustar ganancia para giros a la izquierda
            kp_adjusted = kp
            if error < 0:
                kp_adjusted = kp * 1.5
                if error < -0.3:
                    kp_adjusted = kp * 2.0
            
            # Calcular omega
            omega_cmd = -kp_adjusted * error
            omega_clip = float(np.clip(omega_cmd, -max_omega, max_omega))
            
            # Ajustar suavizado para giros a la izquierda
            alpha_adjusted = alpha
            if error < 0:
                alpha_adjusted = min(0.6, alpha * 2)
                
            omega = alpha_adjusted * omega_clip + (1 - alpha_adjusted) * self.prev_omega
            
        self.prev_omega = omega
        return omega

    def get_trajectory(self):
        return self.traj

    def reset_odometry(self):
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_theta = 0.0
        self.traj = []
        self.prev_omega = 0.0 