from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTrackerV3:
    def __init__(self, max_disappeared=50, max_distance=250):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.velocities = OrderedDict()
        self.disappeared = OrderedDict()
        self.trajectories = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def _predict_position(self, objectID):
        centroid = self.objects[objectID]
        velocity = self.velocities.get(objectID, np.array([0, 0]))
        return centroid + velocity

    def _update_velocity(self, objectID, new_centroid):
        if len(self.trajectories[objectID]) >= 2:
            old_pos = self.trajectories[objectID][-1]
            velocity = new_centroid - old_pos
            self.velocities[objectID] = 0.7 * self.velocities.get(objectID, np.array([0, 0])) + 0.3 * velocity
        else:
            self.velocities[objectID] = np.array([0, 0])

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.velocities[self.nextObjectID] = np.array([0, 0])
        self.disappeared[self.nextObjectID] = 0
        self.trajectories[self.nextObjectID] = [centroid]
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.velocities[objectID]
        del self.disappeared[objectID]
        del self.trajectories[objectID]

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
            predictedCentroids = np.array([self._predict_position(oid) for oid in objectIDs])
            D = dist.cdist(predictedCentroids, input_centroids)
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
                self._update_velocity(objectID, new_centroid)
                self.objects[objectID] = new_centroid
                self.disappeared[objectID] = 0
                self.trajectories[objectID].append(new_centroid)
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
