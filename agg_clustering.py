import numpy as np
from scipy.spatial.distance import pdist, squareform
from Image import Image

#pairwise distance metric, target clusters, linkage, spatial weight

class AgglomerativeClustering:
    def __init__(self, n_clusters=2, linkage='single', spatial_weight=0.1):
        """
        Initialize the AgglomerativeClustering algorithm for images.
        
        Parameters:
        -----------
        n_clusters : int, default=2
            The number of clusters to find.
        linkage : {'single', 'complete', 'average'}, default='single'
            Which linkage criterion to use.
        use_spatial_features : bool, default=True
            Whether to include spatial coordinates as features.
        spatial_weight : float, default=0.1
            Weight for spatial features relative to intensity features.
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.spatial_weight = spatial_weight
        self.labels_ = None


    def fit(self, image:Image):
        """
        Fit the agglomerative clustering on an image.
        
        Parameters:
        -----------
        image : ndarray
            The input image (grayscale or RGB).
            
        Returns:
        --------
        self : object
            Returns the instance itself.
        """
        h, w = image.image.shape[:2]
        
        # Reshape the image into a feature array of dimension Nx1 or Nx3
        if image.is_RGB(): 
            pixel_features = image.image.reshape(-1, 3)
        else:  
            pixel_features = image.reshape(-1, 1)
        
        # X&Y coordinates are added to the feature array
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        x_coords = x_coords.reshape(-1, 1) * self.spatial_weight
        y_coords = y_coords.reshape(-1, 1) * self.spatial_weight
        features = np.hstack([pixel_features, x_coords, y_coords])

        # Compute pairwise distance matrix
        dist_matrix = squareform(pdist(features))
        
        # Perform clustering (reuse existing logic)
        n_samples = features.shape[0]
        clusters = [[i] for i in range(n_samples)]
        while len(clusters) > self.n_clusters:
            min_dist = float('inf')
            merge_i, merge_j = 0, 0
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    dist = self._get_cluster_distance(dist_matrix, clusters[i], clusters[j])
                    if dist < min_dist:
                        min_dist = dist
                        merge_i, merge_j = i, j
            clusters[merge_i].extend(clusters[merge_j])
            clusters.pop(merge_j)
        
        # Assign cluster labels
        self.labels_ = np.zeros(n_samples, dtype=int)
        for i, cluster in enumerate(clusters):
            for idx in cluster:
                self.labels_[idx] = i


    def fit_predict(self, image:Image):
        """
        Fit the agglomerative clustering and return cluster labels.
        
        Parameters:
        -----------
        image : ndarray
            The input image (grayscale or RGB).
            
        Returns:
        --------
        labels : ndarray
            Cluster labels for each pixel.
        """
        self.fit(image)
        return self.labels_.reshape(image.shape[:2])


    def _get_cluster_distance(self, dist_matrix, cluster_i, cluster_j):
        """
        Compute the distance between two clusters based on the linkage criterion.
        
        Parameters:
        -----------
        dist_matrix : ndarray
            The distance matrix.
        cluster_i : list
            Indices of points in the first cluster.
        cluster_j : list
            Indices of points in the second cluster.
            
        Returns:
        --------
        float : The distance between the two clusters.
        """
        if self.linkage == 'single':
            # Minimum distance between any point in cluster_i and any point in cluster_j
            return np.min(dist_matrix[np.ix_(cluster_i, cluster_j)])
        
        elif self.linkage == 'complete':
            # Maximum distance between any point in cluster_i and any point in cluster_j
            return np.max(dist_matrix[np.ix_(cluster_i, cluster_j)])
        
        elif self.linkage == 'average':
            # Average distance between all points in cluster_i and cluster_j
            return np.mean(dist_matrix[np.ix_(cluster_i, cluster_j)])
        
        else:
            raise ValueError(f"Unknown linkage criterion: {self.linkage}")


def segment_image_agg(image: Image, n_clusters=5, linkage='average', spatial_weight=0.1):
    """
    Segment an image using agglomerative clustering.
    
    Parameters:
    -----------
    image_path : str
        Path to the image file.
    n_clusters : int, default=5
        Number of segments to create.
    linkage : str, default='average'
        Linkage criterion to use.
    spatial_weight : float, default=0.1
        Weight for spatial features relative to intensity features.
        
    Returns:
    --------
    segmented_image : ndarray
        The segmented image.
    """    
    # Initialize and fit the clustering
    agglo = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, spatial_weight=spatial_weight)
    labels = agglo.fit_predict(image)
        
    # Create a segmented image
    h, w = image.image.shape[:2]
    segmented_image = np.zeros_like(image.image)
    if not image.is_RGB():  # Grayscale
        unique_labels = np.unique(labels)
        for i, label in enumerate(unique_labels):
            segmented_image[labels == label] = int(255 * (i / len(unique_labels)))
    else:  # RGB
        colors = np.random.randint(0, 255, (n_clusters, 3))
        segmented_image = colors[labels]
    
    return segmented_image