import cv2
import numpy as np

class TrajectoryVisualizer:
    def __init__(self, canvas_size=300, scale=50):
        self.canvas_size = canvas_size
        self.scale = scale  # píxeles por metro
        self.canvas_center = canvas_size // 2
        self.traj_canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255

    def draw_trajectory(self, trajectory):
        # Limpiar lienzo
        self.traj_canvas[:] = 255
        
        if len(trajectory) < 2:
            return self.traj_canvas
            
        # Convertir puntos de odometría a coordenadas de píxeles
        pts = []
        for x, y in trajectory:
            px = int(self.canvas_center + x * self.scale)
            py = int(self.canvas_center - y * self.scale)
            pts.append((px, py))
            
        # Dibujar líneas entre puntos consecutivos
        for i in range(1, len(pts)):
            cv2.line(self.traj_canvas, pts[i-1], pts[i], (255, 0, 0), 2)
            
        return self.traj_canvas

    def reset_canvas(self):
        self.traj_canvas = np.ones((self.canvas_size, self.canvas_size, 3), dtype=np.uint8) * 255 