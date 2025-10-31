from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTrackerV1:
    def __init__(self, max_disappeared=60, max_distance=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance  # Distància màxima per associar centroides

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects, frame):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x, y, w, h)) in enumerate(rects):
            cX = int(x + w / 2.0)
            cY = int(y + h / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = dist.cdist(np.array(objectCentroids), input_centroids)
            
            # Trobar les associacions mínimes que estan dins de la distància màxima
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows, usedCols = set(), set()
            for (row, col) in zip(rows, cols):
                # Només associar si la distància és menor que max_distance
                if row in usedRows or col in usedCols:
                    continue
                
                if D[row, col] <= self.max_distance:
                    self.objects[objectIDs[row]] = input_centroids[col]
                    self.disappeared[objectIDs[row]] = 0
                    usedRows.add(row)
                    usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            for row in unusedRows:
                self.disappeared[objectIDs[row]] += 1
                if self.disappeared[objectIDs[row]] > self.max_disappeared:
                    self.deregister(objectIDs[row])

            for col in unusedCols:
                self.register(input_centroids[col])
        return self.objects