import json
import os
from statistics import median
import json
import os
from statistics import median
from sklearn.cluster import KMeans
import numpy as np


class LabelTracker:
    def __init__(self, data_file='label_data.json'):
        self.data_file = data_file
        self.data = self._load_data()

    def _load_data(self):
        if not os.path.exists(self.data_file):
            with open(self.data_file, 'w') as f:
                json.dump({}, f)

        with open(self.data_file, 'r') as f:
            return json.load(f)

    def _save_data(self):
        with open(self.data_file, 'w') as f:
            json.dump(self.data, f)

    def _update_statistics(self, label, number):
        if label not in self.data:
            self.data[label] = {'min': number, 'max': number, 'sum': number, 'count': 1, 'numbers': [number]}
        else:
            self.data[label]['min'] = min(self.data[label]['min'], number)
            self.data[label]['max'] = max(self.data[label]['max'], number)
            self.data[label]['sum'] += number
            self.data[label]['count'] += 1
            self.data[label]['numbers'].append(number)

        self.data[label]['average'] = self.data[label]['sum'] / self.data[label]['count']
        self.data[label]['median'] = median(self.data[label]['numbers'])

    def track(self, label, number):
        self._update_statistics(label, number)
        self._save_data()
        return self.get_statistics(label)

    def get_statistics(self, label):
        if label in self.data:
            return {
                'min': self.data[label]['min'],
                'max': self.data[label]['max'],
                'median': self.data[label]['median'],
                'average': self.data[label]['average'],
                'count': self.data[label]['count'],
            }
        else:
            return None

    def get_clusters(self, n_clusters=3):
        labels = list(self.data.keys())
        averages = [self.data[label]['average'] for label in labels]
        if not averages:
            return []

        X = np.array(averages).reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
        sorted_cluster_indices = np.argsort(kmeans.cluster_centers_.flatten())

        clusters = {}
        for i, cluster_idx in enumerate(sorted_cluster_indices):
            cluster_name = ['lowest', 'low', 'middle', 'high', 'highest'][i]
            clusters[cluster_name] = [
                label
                for label, label_cluster_idx in zip(labels, kmeans.labels_)
                if label_cluster_idx == cluster_idx
            ]

        return clusters
