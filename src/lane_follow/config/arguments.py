import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Seguimiento de carriles de Duckiebot con integración de odometría')
    parser.add_argument('--map-name', default='udem1', help='Nombre del mapa de Duckietown')
    parser.add_argument('--distortion', action='store_true', help='Aplicar distorsión de cámara')
    parser.add_argument('--draw-curve', action='store_true', help='Dibujar curva de carril')
    parser.add_argument('--draw-bbox', action='store_true', help='Dibujar cuadros delimitadores')
    parser.add_argument('--domain-rand', action='store_true', help='Habilitar aleatorización de dominio')
    parser.add_argument('--frame-skip', default=1, type=int, help='Cuadros a omitir')
    parser.add_argument('--seed', default=1, type=int, help='Semilla aleatoria')
    parser.add_argument('--speed', default=0.2, type=float, help='Velocidad lineal base (m/s)')
    parser.add_argument('--kp', default=1.0, type=float, help='Ganancia proporcional para PID de dirección')
    parser.add_argument('--alpha', default=0.5, type=float, help='Factor de suavizado para omega')
    parser.add_argument('--max-omega', default=1.0, type=float, help='Máxima velocidad angular (rad/s)')
    parser.add_argument('--kernel-size', default=5, type=int, help='Tamaño del núcleo morfológico')
    parser.add_argument('--sample-count', default=20, type=int, help='Número de filas muestreadas para la ruta')
    parser.add_argument('--min-samples', default=3, type=int,
                        help='Número mínimo de puntos requeridos para ajuste polinómico')
    parser.add_argument('--lookahead-dist', default=0.5, type=float, help='Fracción de anticipación a lo largo de la ruta')
    parser.add_argument('--scale', default=50, type=int, help='Píxeles por metro para dibujar la trayectoria')
    parser.add_argument('--canvas-size', default=300, type=int, help='Tamaño del lienzo cuadrado de trayectoria')
    parser.add_argument('--cooldown-duration', default=20, type=int,
                        help='Número de cuadros a esperar antes de detectar otra intersección')
    parser.add_argument('--offset-ratio', default=0.4, type=float,
                        help='Fracción del ancho de la imagen para desplazar la detección a la izquierda (0-1)')
    parser.add_argument('--red-threshold', default=20000, type=int,
                        help='Umbral de conteo de píxeles para detectar franjas rojas de intersección')
    parser.add_argument('--intersection-speed', default=0.1, type=float,
                        help='Velocidad durante el cruce de intersección')
    parser.add_argument('--turn-omega', default=0.85, type=float,
                        help='Velocidad angular durante el giro en intersección')
    parser.add_argument('--frames-to-wait', default=20, type=int,
                        help='Número de cuadros a esperar después del pico rojo antes de girar')
    parser.add_argument('--turn-duration', default=17, type=int,
                        help='Duración del giro en cuadros')
    parser.add_argument('--route-file', default='route_4way.json', type=str,
                        help='Archivo JSON que contiene una ruta predefinida de giros')
    return parser.parse_args() 