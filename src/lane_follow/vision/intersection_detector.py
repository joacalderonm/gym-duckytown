import cv2
import numpy as np

class IntersectionDetector:
    def __init__(self, kernel_size=5, red_threshold=20000, red_history_size=10):
        # Límites HSV para franjas rojas
        self.red_lower1 = np.array([0, 100, 100])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([160, 100, 100])
        self.red_upper2 = np.array([180, 255, 255])
        
        # Parámetros de detección
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.red_threshold = red_threshold
        self.red_history_size = red_history_size
        
        # Estado de detección
        self.red_count_history = []
        self.red_peak_detected = False
        self.frames_after_peak = 0
        self.intersection_cooldown = 0

    def process_frame(self, frame):
        # Convertir a HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        
        # Detección de rojo
        red_mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        red_mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Operaciones morfológicas
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, self.kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, self.kernel)
        
        # Contar píxeles rojos
        red_count = cv2.countNonZero(red_mask)
        
        # Actualizar historial
        self.red_count_history.append(red_count)
        if len(self.red_count_history) > self.red_history_size:
            self.red_count_history.pop(0)
            
        # Detectar pico
        if len(self.red_count_history) >= 3:
            if not self.red_peak_detected and self.red_count_history[-1] < self.red_count_history[-2]:
                self.red_peak_detected = True
                
        # Actualizar contador después del pico
        if self.red_peak_detected:
            self.frames_after_peak += 1
            
        # Actualizar enfriamiento
        if self.intersection_cooldown > 0:
            self.intersection_cooldown -= 1
            
        return red_count, red_mask
        
    def is_intersection_detected(self):
        return (len(self.red_count_history) > 0 and 
                self.red_count_history[-1] > self.red_threshold and 
                self.intersection_cooldown == 0)
                
    def reset_peak_detection(self):
        self.red_peak_detected = False
        self.frames_after_peak = 0
        
    def set_cooldown(self, duration):
        self.intersection_cooldown = duration 