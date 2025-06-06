import cv2
import numpy as np

class TrajectoryDrawer:
    def __init__(self, canvas_size, scale):
        self.canvas_size = canvas_size
        self.scale = scale
        self.traj_canvas = np.ones((self.canvas_size, self.canvas_size, 3), dtype=np.uint8) * 255  # Lienzo blanco
        self.canvas_center = self.canvas_size // 2  # Centro del lienzo
        self.traj = []  # Lista para almacenar la trayectoria del robot (puntos (x, y))

    def update_trajectory(self, x, y):
        """Actualizar la trayectoria del robot con la nueva posición."""
        self.traj.append((x, y))

    def draw_trajectory(self):
        """Dibuja la trayectoria odométrica en un lienzo."""
        self.traj_canvas[:] = 255  # Limpiar el lienzo
        if len(self.traj) < 2:
            return
        pts = []
        for x, y in self.traj:
            px = int(self.canvas_center + x * self.scale)
            py = int(self.canvas_center - y * self.scale)
            pts.append((px, py))
        for i in range(1, len(pts)):
            cv2.line(self.traj_canvas, pts[i-1], pts[i], (255, 0, 0), 2)  # Dibujar líneas entre puntos

    def get_canvas(self):
        """Retorna el lienzo actual con la trayectoria."""
        return self.traj_canvas
