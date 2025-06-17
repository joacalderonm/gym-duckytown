import argparse
from .core.simulator import LaneFollowSimulator

def main():
    parser = argparse.ArgumentParser(description='Simulador de seguimiento de carriles con Duckiebot')
    
    # Parámetros del entorno
    parser.add_argument('--map-name', type=str, default='udem1',
                      help='Nombre del mapa a usar')
    parser.add_argument('--seed', type=int, default=1,
                      help='Semilla aleatoria')
    parser.add_argument('--draw-bbox', action='store_true',
                      help='Dibujar bounding boxes')
    parser.add_argument('--draw-curve', action='store_true',
                      help='Dibujar curvas')
    parser.add_argument('--domain-rand', action='store_true',
                      help='Usar randomización de dominio')
    parser.add_argument('--frame-skip', type=int, default=1,
                      help='Frames a saltar')
    parser.add_argument('--distortion', action='store_true',
                      help='Aplicar distorsión')
    
    # Parámetros de detección de carriles
    parser.add_argument('--kernel-size', type=int, default=5,
                      help='Tamaño del kernel para filtros')
    parser.add_argument('--sample-count', type=int, default=10,
                      help='Número de muestras para detección de carriles')
    parser.add_argument('--min-samples', type=int, default=5,
                      help='Mínimo de muestras para considerar un carril válido')
    parser.add_argument('--offset-ratio', type=float, default=0.5,
                      help='Ratio de offset para detección de carriles')
    parser.add_argument('--white-threshold', type=int, default=200,
                      help='Umbral para detección de líneas blancas')
    
    # Parámetros de detección de intersecciones
    parser.add_argument('--red-threshold', type=int, default=100,
                      help='Umbral para detección de rojo')
    
    # Parámetros de control
    parser.add_argument('--speed', type=float, default=0.44,
                      help='Velocidad base del robot')
    parser.add_argument('--lookahead-dist', type=float, default=0.1,
                      help='Distancia de anticipación para control')
    parser.add_argument('--intersection-speed', type=float, default=0.2,
                      help='Velocidad en intersecciones')
    parser.add_argument('--turn-omega', type=float, default=2.0,
                      help='Velocidad angular para giros')
    parser.add_argument('--frames-to-wait', type=int, default=10,
                      help='Frames a esperar en intersecciones')
    parser.add_argument('--turn-duration', type=int, default=30,
                      help='Duración de giros en frames')
    parser.add_argument('--cooldown-duration', type=int, default=30,
                      help='Duración del cooldown después de giros')
    
    # Parámetros de control PID
    parser.add_argument('--kp', type=float, default=1.0,
                      help='Ganancia proporcional')
    parser.add_argument('--alpha', type=float, default=0.5,
                      help='Factor de suavizado')
    parser.add_argument('--max-omega', type=float, default=2.0,
                      help='Velocidad angular máxima')
    
    # Parámetros de DBSCAN
    parser.add_argument('--dbscan-eps', type=float, default=0.1,
                      help='Radio de vecindad para DBSCAN')
    parser.add_argument('--dbscan-min-samples', type=int, default=5,
                      help='Mínimo de muestras para formar un cluster en DBSCAN')
    
    # Parámetros de evasión de obstáculos
    parser.add_argument('--min-safe-distance', type=float, default=0.3,
                      help='Distancia mínima segura a obstáculos')
    parser.add_argument('--obstacle-speed-factor', type=float, default=0.5,
                      help='Factor de reducción de velocidad para obstáculos')
    
    # Parámetros de visualización
    parser.add_argument('--canvas-size', type=int, default=400,
                      help='Tamaño del canvas para visualización')
    parser.add_argument('--scale', type=float, default=100.0,
                      help='Escala para visualización')
    
    # Parámetros de ruta
    parser.add_argument('--route-file', type=str,
                      help='Archivo JSON con la ruta a seguir')
    
    args = parser.parse_args()
    
    # Crear y ejecutar simulador
    simulator = LaneFollowSimulator(args)
    simulator.run()

if __name__ == '__main__':
    main() 