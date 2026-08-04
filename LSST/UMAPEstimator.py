### Default RAIL imports
from rail.core.stage import RailStage
from rail.core.data import PqHandle, ModelHandle, QPHandle
from ceci.config import StageParameter
import qp

### Specific imports
import numpy as np
import pandas as pd
from umap import UMAP
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import gaussian_kde

### Define custom weighted Manhattan metric
from numba import njit
@njit
def manhattan_weighted_linear(vec1, vec2):
    
    n_bands = len(vec1) // 2
    
    distance = 0.0
    for i in range(n_bands):
        w         = vec1[i + n_bands]**2 + vec2[i + n_bands]**2
        w         = max(w, 1e-4) ### guard against division by zero
        distance += np.abs(vec1[i] - vec2[i])/w

    return distance

class UMAPEstimator(RailStage):
    
    ### stage name
    name = 'UMAPEstimator'
    
    ### inputs and outputs
    inputs = [("training_photometry",   PqHandle),
              ("training_phot_error",   PqHandle),
              ("training_redshift",     PqHandle),
              ("estimation_photometry", PqHandle),
              ("estimation_phot_error", PqHandle)]
    
    outputs = [("informed_reducer",         ModelHandle),
               ("informed_kNN_regressor",   ModelHandle),
               ("informed_embedding",       PqHandle),
               ("estimated_embedding",      PqHandle),
               ("estimated_photoz_medians", PqHandle),
               ("estimated_photoz_pdfs",    QPHandle)]
    
    ### configuration
    config_options = RailStage.config_options.copy()
    
    config_options.update(dict(
        ### UMAP parameters
        n_neighbors_umap     = StageParameter(int,   80),
        n_components         = StageParameter(int,   3),
        ambient_metric_umap  = StageParameter(str,   'euclidean'),
        n_epochs             = StageParameter(int,   -1),
        learning_rate        = StageParameter(float, 1.0),
        init                 = StageParameter(str,   'spectral'),
        min_dist             = StageParameter(float, 0.0),
        spread               = StageParameter(float, 1.0),
        low_memory           = StageParameter(bool,  True),
        set_op_mix_ratio     = StageParameter(float, 1.0),
        local_connectivity   = StageParameter(int,   1),
        repulsion_strength   = StageParameter(float, 1.0),
        negative_sample_rate = StageParameter(int,   5),
        transform_queue_size = StageParameter(float, 4.0),
        metric_kwds          = StageParameter(dict,  {}),
        target_n_neighbors   = StageParameter(int,   -1),
        target_metric_umap   = StageParameter(str,   'categorical'),
        target_metric_kwds   = StageParameter(dict,  {}),
        target_weight        = StageParameter(float, 0.5),
        transform_seed       = StageParameter(int,   42),
        
        ### k-nearest neighbor clustering parameters
        n_neighbors_knn      = StageParameter(int, 10),
        metric_knn           = StageParameter(str, 'minkowski'),
        metric_p_knn         = StageParameter(int, 2),
        weighting_knn        = StageParameter(str, 'distance'),
        algorithm_knn        = StageParameter(str, 'ball_tree'),
        
        ### gaussian kernel density estimator parameters
        # weighting_gauss_kde  = StageParameter(str, 'inverse_sq_dist'),
        bw_method_gauss_kde  = StageParameter(str, 'silverman'),
        zmax_gauss_kde       = StageParameter(float, 10.0),
        precision_gauss_kde  = StageParameter(float, 0.001),
        
        
        ### random state
        random_state         = StageParameter(int, 42)
        ))
 
 
    def UMAP_informer(self):
        
        photometry = self.get_data("training_photometry")
        
        input_data = photometry
        
        if self.config.ambient_metric_umap == 'manhattan_weighted_linear':
            
            phot_error = self.get_data("training_phot_error")
            input_data = pd.concat([photometry, phot_error], axis = 1)
            metric = manhattan_weighted_linear
        
        else:
            metric = self.config.ambient_metric_umap
        
        n_epochs           = None if self.config.n_epochs                == -1 else self.config.n_epochs
        metric_kwds        = None if len(self.config.metric_kwds)        == 0  else self.config.metric_kwds
        target_metric_kwds = None if len(self.config.target_metric_kwds) == 0 else self.config.target_metric_kwds
        
        reducer = UMAP(
            metric               =             metric, 
            n_neighbors          = self.config.n_neighbors_umap, 
            n_components         = self.config.n_components, 
            n_epochs             =             n_epochs, 
            learning_rate        = self.config.learning_rate, 
            init                 = self.config.init, 
            min_dist             = self.config.min_dist, 
            spread               = self.config.spread, 
            low_memory           = self.config.low_memory, 
            set_op_mix_ratio     = self.config.set_op_mix_ratio, 
            local_connectivity   = self.config.local_connectivity, 
            repulsion_strength   = self.config.repulsion_strength, 
            negative_sample_rate = self.config.negative_sample_rate, 
            transform_queue_size = self.config.transform_queue_size, 
            metric_kwds          =             metric_kwds, 
            target_n_neighbors   = self.config.target_n_neighbors, 
            target_metric        = self.config.target_metric_umap, 
            target_metric_kwds   =             target_metric_kwds, 
            target_weight        = self.config.target_weight, 
            transform_seed       = self.config.transform_seed, 
            random_state         = self.config.random_state
            )
        
        embedding = reducer.fit_transform(input_data)
        
        self.add_data("informed_reducer",   reducer)
        self.add_data("informed_embedding", pd.DataFrame(embedding))
        
        redshift = self.get_data("training_redshift")
        
        kNN_regressor = KNeighborsRegressor(
            n_neighbors = self.config.n_neighbors_knn,
            metric      = self.config.metric_knn,
            p           = self.config.metric_p_knn,
            weights     = self.config.weighting_knn,
            algorithm   = self.config.algorithm_knn,
        )
        
        kNN_regressor.fit(embedding, redshift)
        
        informed_kNN_regressor = dict(
            model = kNN_regressor,
            redshift = redshift
        )
        
        self.add_data("informed_kNN_regressor", informed_kNN_regressor)
        
    
    def UMAP_estimator(self):
        
        reducer = self.get_data("informed_reducer")
        
        photometry = self.get_data("estimation_photometry")
        
        if reducer.metric == manhattan_weighted_linear:
            
            phot_error = self.get_data("estimation_phot_error")
            input_data = pd.concat([photometry, phot_error], axis = 1)    
        
        else:
            input_data = photometry
        
        estimated_embedding = reducer.transform(input_data)
        
        informed_kNN_regressor = self.get_data("informed_kNN_regressor")
        all_distances, all_indices = informed_kNN_regressor["model"].kneighbors(estimated_embedding)
        
        z_grid = np.arange(0, self.config.zmax_gauss_kde, self.config.precision_gauss_kde)
        
        def generate_pdf(data_points, distances):
            
            kde = gaussian_kde(data_points,
                               weights = 1/np.maximum(distances**2, 1e-10),
                               bw_method = self.config.bw_method_gauss_kde)
            
            return kde(z_grid)
        
        photoz_pdfs = np.array(
            [generate_pdf(informed_kNN_regressor["redshift"][indices], distances)
             for indices, distances in zip(all_indices, all_distances)])
        photoz_cdfs = np.cumsum(photoz_pdfs, axis = -1)
        photoz_cdfs /= photoz_cdfs[:, -1:]
        photoz_medians = z_grid[np.argmax(photoz_cdfs >= 0.5, axis = 1)]
        
        photoz_pdf_ensemble = qp.Ensemble(qp.interp, data = dict(xvals = z_grid, yvals = photoz_pdfs))
        
        self.add_data("estimated_embedding",        pd.DataFrame(estimated_embedding))
        self.add_data("estimated_photoz_medians",   pd.DataFrame({"z_phot_median": photoz_medians}))
        self.add_data("estimated_photoz_pdfs",      photoz_pdf_ensemble)