class VehicleCounter:
    def __init__(self, line_y):
        self.line_y = line_y
        self.counts = {"north": 0, "south": 0}
        self.track_history = {}

    def update_counts(self, objects):
        for objectID, centroid in objects.items():
            if objectID not in self.track_history:
                self.track_history[objectID] = []
            self.track_history[objectID].append(centroid)

            if len(self.track_history[objectID]) >= 2:
                y_positions = [p[1] for p in self.track_history[objectID][-2:]]
                direction = y_positions[-1] - y_positions[-2]
                if y_positions[-2] < self.line_y and y_positions[-1] >= self.line_y:
                    self.counts["south"] += 1
                elif y_positions[-2] > self.line_y and y_positions[-1] <= self.line_y:
                    self.counts["north"] += 1
        return self.counts
