import numpy as np
from scipy.spatial.distance import pdist, squareform

class AgglomerativeClustering:
    def __init__(self, n_clusters=2, linkage='single', use_spatial_features=True, spatial_weight=0.1):
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
        self.use_spatial_features = use_spatial_features
        self.spatial_weight = spatial_weight
        self.labels_ = None

    def _compute_distance_matrix(self, X):
        """Compute the distance matrix between all pairs of points."""
        return squareform(pdist(X))

    def fit(self, image):
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
        h, w = image.shape[:2]
        
        # Reshape the image into a feature array
        if len(image.shape) == 2:  # Grayscale
            pixel_features = image.reshape(-1, 1)
        else:  # RGB
            pixel_features = image.reshape(-1, 3)
        
        # Add spatial features if enabled
        if self.use_spatial_features:
            y_coords, x_coords = np.mgrid[0:h, 0:w]
            x_coords = x_coords.reshape(-1, 1) * self.spatial_weight
            y_coords = y_coords.reshape(-1, 1) * self.spatial_weight
            features = np.hstack([pixel_features, x_coords, y_coords])
        else:
            features = pixel_features
        
        # Compute pairwise distance matrix
        dist_matrix = self._compute_distance_matrix(features)
        
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
        
        return self

    def fit_predict(self, image):
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


def segment_image_agg(image_arr, n_clusters=5, linkage='average', spatial_weight=0.1):
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
    labels = agglo.fit_predict(image_arr)
    
    # Create a segmented image
    h, w = image_arr.shape[:2]
    segmented_image = np.zeros_like(image_arr)
    if len(image_arr.shape) == 2:  # Grayscale
        unique_labels = np.unique(labels)
        for i, label in enumerate(unique_labels):
            segmented_image[labels == label] = int(255 * (i / len(unique_labels)))
    else:  # RGB
        colors = np.random.randint(0, 255, (n_clusters, 3))
        segmented_image = colors[labels]
    
    return segmented_image