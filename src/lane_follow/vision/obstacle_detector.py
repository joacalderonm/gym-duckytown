import numpy as np
from sklearn.cluster import DBSCAN
import cv2

class ObstacleDetector:
    def __init__(self, eps=30, min_samples=5):
        """
        Initialize the obstacle detector with DBSCAN parameters.
        
        Args:
            eps (float): The maximum distance between two samples for them to be considered neighbors
            min_samples (int): The number of samples in a neighborhood for a point to be considered a core point
        """
        self.eps = eps
        self.min_samples = min_samples
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        
        # Colors for visualization (BGR format)
        self.cluster_colors = [
            (0, 0, 255),    # Red
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Cyan
        ]

    def detect_clusters(self, points):
        """
        Detect clusters in the given points using DBSCAN.
        
        Args:
            points (list): List of (x, y) coordinates
            
        Returns:
            tuple: (labels, cluster_centers, cluster_sizes)
                - labels: Array of cluster labels (-1 for noise points)
                - cluster_centers: List of (x, y) coordinates for cluster centers
                - cluster_sizes: List of sizes for each cluster
        """
        if not points:
            return np.array([]), [], []
            
        # Convert points to numpy array
        points_array = np.array(points)
        
        # Perform DBSCAN clustering
        labels = self.dbscan.fit_predict(points_array)
        
        # Calculate cluster centers and sizes
        unique_labels = np.unique(labels)
        cluster_centers = []
        cluster_sizes = []
        
        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue
                
            # Get points in this cluster
            cluster_points = points_array[labels == label]
            
            # Calculate cluster center
            center = np.mean(cluster_points, axis=0)
            cluster_centers.append(center)
            
            # Calculate cluster size
            size = len(cluster_points)
            cluster_sizes.append(size)
            
        return labels, cluster_centers, cluster_sizes

    def visualize_clusters(self, frame, points, labels, cluster_centers):
        """
        Visualize the detected clusters on the frame.
        
        Args:
            frame: The image frame to draw on
            points: List of (x, y) coordinates
            labels: Array of cluster labels
            cluster_centers: List of cluster center coordinates
            
        Returns:
            The frame with cluster visualization
        """
        vis_frame = frame.copy()
        
        # Draw points with their cluster colors
        for i, (x, y) in enumerate(points):
            if labels[i] == -1:  # Noise points
                cv2.circle(vis_frame, (int(x), int(y)), 3, (128, 128, 128), -1)
            else:
                color = self.cluster_colors[labels[i] % len(self.cluster_colors)]
                cv2.circle(vis_frame, (int(x), int(y)), 3, color, -1)
        
        # Draw cluster centers
        for i, center in enumerate(cluster_centers):
            color = self.cluster_colors[i % len(self.cluster_colors)]
            cv2.circle(vis_frame, (int(center[0]), int(center[1])), 5, color, -1)
            cv2.putText(vis_frame, f'C{i}', 
                       (int(center[0]) + 5, int(center[1]) + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis_frame

    def get_obstacle_risk(self, cluster_sizes, max_size=100):
        """
        Calculate the risk level based on cluster sizes.
        
        Args:
            cluster_sizes: List of cluster sizes
            max_size: Maximum cluster size for normalization
            
        Returns:
            float: Risk level between 0 and 1
        """
        if not cluster_sizes:
            return 0.0
            
        # Calculate normalized risk based on largest cluster
        max_cluster_size = max(cluster_sizes)
        risk = min(1.0, max_cluster_size / max_size)
        
        return risk 