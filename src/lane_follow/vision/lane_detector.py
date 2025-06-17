import cv2
import numpy as np
from sklearn.cluster import DBSCAN

class LaneDetector:
    def __init__(self, kernel_size=5, sample_count=10, min_samples=5, offset_ratio=0.5,
                 dbscan_eps=0.1, dbscan_min_samples=5):
        """
        Inicializa el detector de carriles con DBSCAN.
        
        Args:
            kernel_size: Tamaño del kernel para filtros
            sample_count: Número de muestras para detección de carriles
            min_samples: Mínimo de muestras para considerar un carril válido
            offset_ratio: Ratio de offset para detección de carriles
            dbscan_eps: Radio de vecindad para DBSCAN
            dbscan_min_samples: Mínimo de muestras para formar un cluster en DBSCAN
        """
        self.kernel_size = kernel_size
        self.sample_count = sample_count
        self.min_samples = min_samples
        self.offset_ratio = offset_ratio
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        
        # Inicializar DBSCAN
        self.dbscan = DBSCAN(
            eps=dbscan_eps,
            min_samples=dbscan_min_samples,
            metric='euclidean'
        )
        
        # Historial de puntos para suavizado
        self.point_history = []
        self.history_size = 5
        
        # Definir rangos HSV para cada tipo de línea
        # Rojo (dos rangos debido a la naturaleza circular del espacio HSV)
        self.red_lower1 = np.array([0, 50, 50])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([170, 50, 50])
        self.red_upper2 = np.array([180, 255, 255])
        
        # Amarillo
        self.yellow_lower = np.array([20, 100, 100])
        self.yellow_upper = np.array([35, 255, 255])
        
        # Blanco
        self.white_lower = np.array([0, 0, 70])
        self.white_upper = np.array([90, 20, 255])

    def create_color_masks(self, frame):
        """
        Crea máscaras para cada tipo de línea usando detección de color HSV.
        
        Args:
            frame: Frame de entrada en formato BGR
            
        Returns:
            masks: Diccionario con máscaras para cada tipo de línea
        """
        # Convertir a HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Crear kernel para operaciones morfológicas
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        
        # Detectar líneas rojas (dos rangos)
        red_mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        red_mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Detectar líneas amarillas
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        
        # Detectar líneas blancas
        white_mask = cv2.inRange(hsv, self.white_lower, self.white_upper)
        
        # Aplicar operaciones morfológicas a cada máscara
        masks = {}
        for name, mask in [('red', red_mask), ('yellow', yellow_mask), ('white', white_mask)]:
            # Limpiar ruido
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            # Cerrar huecos
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            masks[name] = mask
        
        return masks

    def detect_lane_points(self, masks):
        """
        Detecta puntos de carril usando DBSCAN en las máscaras de color.
        
        Args:
            masks: Diccionario con máscaras para cada tipo de línea
            
        Returns:
            points: Diccionario con puntos detectados para cada tipo de línea
            labels: Diccionario con etiquetas de clusters para cada tipo de línea
        """
        points = {}
        labels = {}
        
        for color, mask in masks.items():
            # Encontrar contornos en la máscara
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Extraer puntos de los contornos
            color_points = []
            for contour in contours:
                # Aproximar el contorno a una línea
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Agregar puntos al conjunto
                for point in approx:
                    x, y = point[0]
                    color_points.append([x, y])
            
            if not color_points:
                points[color] = []
                labels[color] = []
                continue
                
            # Convertir a array numpy
            color_points = np.array(color_points)
            
            # Aplicar DBSCAN
            color_labels = self.dbscan.fit_predict(color_points)
            
            points[color] = color_points
            labels[color] = color_labels
        
        return points, labels

    def filter_lane_clusters(self, points, labels):
        """
        Filtra los clusters para identificar las líneas de carril.
        
        Args:
            points: Diccionario con puntos detectados
            labels: Diccionario con etiquetas de clusters
            
        Returns:
            lane_points: Puntos filtrados que pertenecen a las líneas de carril
        """
        lane_points = []
        
        for color in ['yellow', 'white']:  # Procesar solo líneas amarillas y blancas
            if color not in points or not points[color].size:
                continue
                
            # Obtener clusters únicos (excluyendo ruido, label=-1)
            unique_labels = set(labels[color])
            if -1 in unique_labels:
                unique_labels.remove(-1)
                
            for label in unique_labels:
                # Obtener puntos del cluster
                cluster_points = points[color][labels[color] == label]
                
                # Calcular características del cluster
                if len(cluster_points) >= self.min_samples:
                    # Calcular la orientación del cluster
                    x_coords = cluster_points[:, 0]
                    y_coords = cluster_points[:, 1]
                    
                    # Ajustar una línea a los puntos
                    if len(x_coords) > 1:
                        slope, _ = np.polyfit(x_coords, y_coords, 1)
                        
                        # Filtrar basado en la orientación (líneas más verticales)
                        if abs(slope) < 0.5:  # Ajustar este umbral según sea necesario
                            lane_points.extend(cluster_points.tolist())
        
        return lane_points

    def process_frame(self, frame, turn_action=None):
        """
        Procesa un frame para detectar carriles usando DBSCAN.
        
        Args:
            frame: Frame de entrada en formato BGR
            turn_action: Acción de giro actual (opcional)
            
        Returns:
            path_points: Puntos del camino detectado
            vis: Frame con visualización
            masks: Diccionario con máscaras de color
        """
        # Crear máscaras de color
        masks = self.create_color_masks(frame)
        
        # Detectar puntos de carril
        points, labels = self.detect_lane_points(masks)
        
        # Filtrar clusters para obtener líneas de carril
        lane_points = self.filter_lane_clusters(points, labels)
        
        # Actualizar historial de puntos
        if lane_points:
            self.point_history.append(lane_points)
            if len(self.point_history) > self.history_size:
                self.point_history.pop(0)
        
        # Suavizar puntos usando el historial
        if self.point_history:
            all_points = [p for points in self.point_history for p in points]
            path_points = self.smooth_points(all_points)
        else:
            path_points = []
        
        # Visualizar resultados
        vis = frame.copy()
        
        # Dibujar máscaras de color
        vis[masks['red'] > 0] = [0, 0, 255]  # Rojo para líneas rojas
        vis[masks['yellow'] > 0] = [0, 255, 255]  # Amarillo para líneas amarillas
        vis[masks['white'] > 0] = [255, 255, 255]  # Blanco para líneas blancas
        
        # Dibujar clusters
        for color in ['red', 'yellow', 'white']:
            if color not in points or not np.array(points[color]).size:
                continue
                
            for label in set(labels[color]):
                if label == -1:
                    continue
                    
                # Obtener puntos del cluster
                cluster_points = points[color][labels[color] == label]
                
                # Dibujar puntos del cluster
                for point in cluster_points:
                    x, y = point
                    if color == 'red':
                        cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)
                    elif color == 'yellow':
                        cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 255), -1)
                    else:  # white
                        cv2.circle(vis, (int(x), int(y)), 2, (255, 255, 255), -1)
                
                # Dibujar centro del cluster
                center = np.mean(cluster_points, axis=0)
                cv2.circle(vis, (int(center[0]), int(center[1])), 5, (0, 255, 0), -1)
        
        # Dibujar camino suavizado
        if path_points:
            for i in range(len(path_points) - 1):
                pt1 = (int(path_points[i][0]), int(path_points[i][1]))
                pt2 = (int(path_points[i+1][0]), int(path_points[i+1][1]))
                cv2.line(vis, pt1, pt2, (255, 255, 0), 2)
        
        return path_points, vis, masks

    def smooth_points(self, points):
        """
        Suaviza los puntos del camino usando un filtro de media móvil.
        
        Args:
            points: Lista de puntos (x,y)
            
        Returns:
            smoothed_points: Puntos suavizados
        """
        if not points:
            return []
            
        # Convertir a array numpy
        points = np.array(points)
        
        # Ordenar puntos por coordenada x
        sort_idx = np.argsort(points[:, 0])
        points = points[sort_idx]
        
        # Aplicar filtro de media móvil
        window_size = min(5, len(points))
        if window_size > 1:
            kernel = np.ones(window_size) / window_size
            smoothed_x = np.convolve(points[:, 0], kernel, mode='valid')
            smoothed_y = np.convolve(points[:, 1], kernel, mode='valid')
            smoothed_points = np.column_stack((smoothed_x, smoothed_y))
        else:
            smoothed_points = points
            
        return smoothed_points.tolist() 