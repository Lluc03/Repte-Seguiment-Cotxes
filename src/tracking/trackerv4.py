from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
from skimage.feature import hog
import cv2

class CentroidTrackerV4:
    def __init__(self, max_disappeared=50, max_distance=250, direction_threshold=10,
                 alpha=0.8, beta=0.2, velocity_weight=0.3):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.trajectories = OrderedDict()
        self.directions = OrderedDict()
        self.hogs = OrderedDict()
        self.velocities = OrderedDict()  # 🆕 historial de velocidades
        self.confidence = OrderedDict()  # 🆕 confianza en cada match
        
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.direction_threshold = direction_threshold
        self.trajectory_length = 10  # 🔧 aumentado de 5 a 10
        
        # Pesos ajustados
        self.alpha = alpha  # distancia centroides (más importante)
        self.beta = beta    # distancia HOG (menos importante en oclusiones)
        self.velocity_weight = velocity_weight  # 🆕 peso para predicción de velocidad

    def _extract_hog(self, frame, rect):
        """Extrae el descriptor HOG de un bounding box"""
        (x, y, w, h) = rect
        # Validar límites
        x, y = max(0, x), max(0, y)
        x2, y2 = min(frame.shape[1], x+w), min(frame.shape[0], y+h)
        
        if x2 <= x or y2 <= y:
            return np.zeros(100)
            
        patch = frame[y:y2, x:x2]
        if patch.size == 0 or patch.shape[0] < 16 or patch.shape[1] < 16:
            return np.zeros(100)
            
        patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        # Redimensionar a tamaño fijo para HOG consistente
        patch_gray = cv2.resize(patch_gray, (64, 64))
        
        try:
            fd = hog(patch_gray, orientations=8, pixels_per_cell=(16,16),
                     cells_per_block=(1,1), feature_vector=True)
        except:
            fd = np.zeros(100)
        return fd

    def register(self, centroid, hog_desc):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.trajectories[self.nextObjectID] = [centroid]
        self.directions[self.nextObjectID] = "UNKNOWN"
        self.hogs[self.nextObjectID] = hog_desc
        self.velocities[self.nextObjectID] = np.array([0, 0])
        self.confidence[self.nextObjectID] = 1.0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.trajectories[objectID]
        del self.directions[objectID]
        del self.hogs[objectID]
        del self.velocities[objectID]
        del self.confidence[objectID]

    def _calculate_direction(self, objectID, new_centroid):
        """Calcula dirección usando toda la trayectoria"""
        trajectory = self.trajectories[objectID]
        if len(trajectory) < 3:  # 🔧 necesitamos más puntos
            return self.directions[objectID] if objectID in self.directions else "UNKNOWN"
        
        # Calcular dirección promedio de los últimos N puntos
        dx_total = trajectory[-1][0] - trajectory[0][0]
        dy_total = trajectory[-1][1] - trajectory[0][1]
        
        if abs(dx_total) > abs(dy_total):
            if dx_total > self.direction_threshold: return "EAST"
            elif dx_total < -self.direction_threshold: return "WEST"
        else:
            if dy_total > self.direction_threshold: return "SOUTH"
            elif dy_total < -self.direction_threshold: return "NORTH"
        return "UNKNOWN"

    def _predict_position(self, objectID):
        """🆕 Predice la siguiente posición basándose en velocidad"""
        if objectID not in self.velocities:
            return self.objects[objectID]
        
        current_pos = self.objects[objectID]
        velocity = self.velocities[objectID]
        predicted = current_pos + velocity
        return predicted.astype(int)

    def _update_velocity(self, objectID, new_centroid):
        """🆕 Actualiza el vector de velocidad"""
        if len(self.trajectories[objectID]) >= 2:
            old_pos = self.trajectories[objectID][-1]
            velocity = new_centroid - old_pos
            # Suavizado exponencial
            if objectID in self.velocities:
                self.velocities[objectID] = 0.7 * self.velocities[objectID] + 0.3 * velocity
            else:
                self.velocities[objectID] = velocity
        else:
            self.velocities[objectID] = np.array([0, 0])

    def update(self, rects, frame):
        """rects = [(x, y, w, h), ...], frame = imagen actual"""
        if len(rects) == 0:
            # 🔧 Manejo mejorado de frames sin detecciones
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                # Reducir confianza gradualmente
                self.confidence[objectID] *= 0.95
                
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        input_hogs = []
        for (i, (x, y, w, h)) in enumerate(rects):
            cX = int(x + w / 2.0)
            cY = int(y + h / 2.0)
            input_centroids[i] = (cX, cY)
            input_hogs.append(self._extract_hog(frame, (x, y, w, h)))

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], input_hogs[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = np.array(list(self.objects.values()))
            objectHogs = [self.hogs[i] for i in objectIDs]

            # 🆕 Predicción de posiciones basada en velocidad
            predicted_centroids = np.array([self._predict_position(oid) for oid in objectIDs])

            # Distancia a posición actual
            D_centroid_current = dist.cdist(objectCentroids, input_centroids)
            
            # 🆕 Distancia a posición predicha (ayuda en oclusiones)
            D_centroid_predicted = dist.cdist(predicted_centroids, input_centroids)
            
            # Combinar ambas distancias
            D_centroid = 0.6 * D_centroid_current + 0.4 * D_centroid_predicted

            # Calcular distancias HOG con manejo de errores
            D_hog = np.ones((len(objectHogs), len(input_hogs)))  # Default: máxima distancia
            for i, h1 in enumerate(objectHogs):
                for j, h2 in enumerate(input_hogs):
                    if len(h1) == len(h2) and len(h1) > 0:
                        try:
                            # 🔧 Usar distancia euclidiana normalizada (más robusta que cosine)
                            D_hog[i, j] = np.linalg.norm(h1 - h2) / (np.linalg.norm(h1) + np.linalg.norm(h2) + 1e-5)
                        except:
                            D_hog[i, j] = 1.0

            # 🔧 Normalizar distancias HOG para que estén en rango similar a D_centroid
            if D_hog.max() > 0:
                D_hog = D_hog / D_hog.max() * D_centroid.max()

            # Combinación ponderada con ajuste dinámico
            # 🆕 Reducir peso de HOG cuando hay alta probabilidad de oclusión
            confidence_matrix = np.array([[self.confidence[oid] for _ in input_centroids] 
                                         for oid in objectIDs])
            
            # Cuando la confianza es baja (posible oclusión), priorizar distancia espacial
            adaptive_beta = self.beta * confidence_matrix
            adaptive_alpha = self.alpha + (self.beta - adaptive_beta)
            
            D_total = adaptive_alpha * D_centroid + adaptive_beta * D_hog

            # 🆕 Penalización por cambio de dirección abrupto
            for i, oid in enumerate(objectIDs):
                if self.directions[oid] != "UNKNOWN":
                    for j, new_centroid in enumerate(input_centroids):
                        temp_traj = self.trajectories[oid] + [new_centroid]
                        dx = temp_traj[-1][0] - temp_traj[-2][0]
                        
                        # Penalizar si cambia de dirección horizontal abruptamente
                        if self.directions[oid] == "EAST" and dx < -10:
                            D_total[i, j] *= 1.5
                        elif self.directions[oid] == "WEST" and dx > 10:
                            D_total[i, j] *= 1.5

            rows = D_total.min(axis=1).argsort()
            cols = D_total.argmin(axis=1)[rows]
            usedRows, usedCols = set(), set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                    
                objectID = objectIDs[row]
                distance = D_total[row, col]
                
                # 🔧 Umbral adaptativo: más permisivo si el objeto ha desaparecido recientemente
                adaptive_max_distance = self.max_distance * (1 + 0.5 * (self.disappeared[objectID] / self.max_disappeared))
                
                if distance <= adaptive_max_distance:
                    new_centroid = input_centroids[col]
                    self.objects[objectID] = new_centroid
                    self.disappeared[objectID] = 0
                    
                    # 🆕 Actualizar velocidad antes de agregar a trayectoria
                    self._update_velocity(objectID, new_centroid)
                    
                    self.trajectories[objectID].append(new_centroid)
                    if len(self.trajectories[objectID]) > self.trajectory_length:
                        self.trajectories[objectID].pop(0)
                    
                    self.directions[objectID] = self._calculate_direction(objectID, new_centroid)
                    
                    # 🆕 Actualizar HOG solo si la confianza es alta (evita contaminar con oclusiones)
                    if distance < self.max_distance * 0.7:
                        self.hogs[objectID] = input_hogs[col]
                        self.confidence[objectID] = min(1.0, self.confidence[objectID] + 0.1)
                    else:
                        # Si el match es dudoso, mantener HOG anterior
                        self.confidence[objectID] *= 0.9
                    
                    usedRows.add(row)
                    usedCols.add(col)

            # Manejar objetos no asignados
            unusedRows = set(range(D_total.shape[0])).difference(usedRows)
            unusedCols = set(range(D_total.shape[1])).difference(usedCols)
            
            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                self.confidence[objectID] *= 0.95
                
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)
                    
            for col in unusedCols:
                self.register(input_centroids[col], input_hogs[col])

        return self.objects