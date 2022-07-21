class KMeansClustering():
  import numpy as np
  def __init__(self, n_clusters, X, n_iters):
    self.n_clusters = n_clusters
    self.X = X
    self.n_iters = n_iters

  def fit(self):
    # Initialize random centroids by selecting random values from the dataset
    # Shape will be: (n_features x n_clusters)
    centroids = [self.X[random.randrange(0,len(self.X))] for cluster in range(self.n_clusters)]

    # Perform n simulations until convergence 
    for _ in range(self.n_iters):
      # Initialize n amount of clusters which datapoints will be appended 
      clusters = [[] for x in range(self.n_clusters)]

      for idx, data_point in enumerate(self.X): 
        prospective_centroids = []
        for centroid in centroids:
          x = 0 
          for dt,ct in zip(data_point,centroid):
            # Calculate the Euclidean distance 
            x += (dt - ct) ** 2 
          x = x ** 0.5 
          prospective_centroids.append(x)
        # Index the closest centroid 
        closest_centroid_idx = prospective_centroids.index(min(prospective_centroids))
        # Append each datapoint to its respective cluster  
        clusters[closest_centroid_idx].append(idx)

      # Update centroids with the mean of each previously classified cluster
      # mean is calculated on feature-level 
      new_centroids = [[self.X[i] for i in cluster] for cluster in clusters]
      for i,_ in enumerate(centroids):
        centroids[i] = np.mean(new_centroids[i], axis=0)
    
    return (clusters, centroids)
