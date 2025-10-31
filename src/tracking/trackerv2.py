from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTrackerV2:
    def __init__(self, max_disappeared=50, max_distance=250, direction_threshold=10):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.trajectories = OrderedDict()
        self.directions = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.direction_threshold = direction_threshold

    def _calculate_direction(self, objectID):
        trajectory = self.trajectories[objectID]
        if len(trajectory) < 2:
            return "UNKNOWN"
        dx = trajectory[-1][0] - trajectory[0][0]
        dy = trajectory[-1][1] - trajectory[0][1]
        if abs(dx) > abs(dy):
            return "EAST" if dx > self.direction_threshold else "WEST" if dx < -self.direction_threshold else "UNKNOWN"
        else:
            return "SOUTH" if dy > self.direction_threshold else "NORTH" if dy < -self.direction_threshold else "UNKNOWN"

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.trajectories[self.nextObjectID] = [centroid]
        self.directions[self.nextObjectID] = "UNKNOWN"
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.trajectories[objectID]
        del self.directions[objectID]

    def update(self, rects, frame=None):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x, y, w, h)) in enumerate(rects):
            input_centroids[i] = (int(x + w / 2), int(y + h / 2))

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = np.array(list(self.objects.values()))
            D = dist.cdist(objectCentroids, input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows, usedCols = set(), set()
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                if D[row, col] > self.max_distance:
                    continue
                objectID = objectIDs[row]
                new_centroid = input_centroids[col]
                self.objects[objectID] = new_centroid
                self.disappeared[objectID] = 0
                self.trajectories[objectID].append(new_centroid)
                self.directions[objectID] = self._calculate_direction(objectID)
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(D.shape[0])).difference(usedRows)
            unusedCols = set(range(D.shape[1])).difference(usedCols)
            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)
            for col in unusedCols:
                self.register(input_centroids[col])

        return self.objects
