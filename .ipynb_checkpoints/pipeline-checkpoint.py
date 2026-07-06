# ----------------------------------------------------------------------------------------------- #
# Estimating the impact of using photo-zs in place of spec-zs when estimating photo-zs with UMAPs #
# Author: Leonid Sajkov (sajkov@stanford.edu)                                                     #
# July 2026                                                                                       #
# ----------------------------------------------------------------------------------------------- #

# ----------------------------------------------------------------------------------------------- #
# Section 0: Setup                                                                                #
# ----------------------------------------------------------------------------------------------- #

print("Starting pipeline.")

### Set the date, start the timer
import time
date = time.strftime('%d%b%y', time.localtime())

def timestamp():
    return(f"[{time.strftime("%H:%M:%S", time.localtime())}]")

def timer(start_time, return_tuple = False):
    elapsed_time = time.time() - start_time

    hours = elapsed_time//3600
    minutes = (elapsed_time//60)%60
    seconds = elapsed_time - 3600 * hours - 60 * minutes
    
    if return_tuple:
        return (hours, minutes, seconds)
    
    if hours > 0:
        return f'{hours:.0f}h {minutes:.0f}m {seconds:.1f}s'
    
    if minutes > 0:
        return f'{minutes:.0f}m {seconds:.1f}s'
    
    return f'{seconds:.1f}s'

start_time = time.time()

print(timestamp(), "Started timer. Importing modules", end = "\r")

### System imports
import sys, os, shutil
import glob

### I/O imports
import h5py
import tables_io
import pandas as pd

### Operations
import numpy as np

### RAIL imports
from rail.core.data import DataStore, PqHandle, ModelHandle
from rail.core.stage import RailStage

from MultiSurveyErrorModel import MultiSurveyErrorModel
from UMAPEstimator import UMAPEstimator

print(timestamp(), "Finished importing modules. Time elapsed: ", timer(start_time))

print(timestamp(), "Setting up LePhare", end = "\r")

### Set up LePhare
path_to_lp_config_file    = "/pscratch/sd/s/sajkov/analysis_pipeline/lephare/lsst.para"
os.environ["LEPHAREDIR"]  = f"{os.path.dirname(path_to_lp_config_file)}/data"
os.environ["LEPHAREWORK"] = f"{os.path.dirname(path_to_lp_config_file)}/work"

from rail.estimation.algos.lephare import LephareInformer, LephareEstimator
import lephare as lp
lephare_config = lp.read_config("/pscratch/sd/s/sajkov/analysis_pipeline/lephare/lsst.para")
lp.data_retrieval.get_auxiliary_data(keymap = lephare_config)

path_to_filters = "/pscratch/sd/s/sajkov/analysis_pipeline/filters"
os.makedirs(f"{os.environ['LEPHAREDIR']}/filt/pipeline", exist_ok = True)
for f in glob.glob(f"{path_to_filters}/*.dat"):
    if f.endswith("F146.dat"):
        continue
    shutil.copy(f, f"{os.environ['LEPHAREDIR']}/filt/pipeline/")

FILTER_LIST = ",".join([f"pipeline/{os.path.basename(f)}" for f in glob.glob(f"{os.environ['LEPHAREDIR']}/filt/pipeline/*.dat")])
lephare_config["FILTER_LIST"].value = FILTER_LIST
lephare_config["FILTER_FILE"].value = "filter_pipeline"

os.makedirs(f"{os.environ['LEPHAREDIR']}/output", exist_ok = True)
lephare_config["PARA_OUT"].value = f"{os.environ['LEPHAREDIR']}/output/output_{date}.para"

lp.data_retrieval.get_auxiliary_data(keymap=lephare_config)

print(timestamp(), "Set up LePhare. Time elapsed: ", timer(start_time))

### Initialize random state
seed = 42
rng = np.random.default_rng(seed = seed)

### Specify outputs directory
outputs_directory = f"/pscratch/sd/s/sajkov/analysis_pipeline/runs/{date}"
os.makedirs(outputs_directory, exist_ok = True)
    
# ----------------------------------------------------------------------------------------------- #
# Section 1: Create datasets                                                                      #
# ----------------------------------------------------------------------------------------------- #

print(timestamp(), "Creating datasets.", end = "\r")

### Specify path to noiseless catalog and redshifts
noiseless_catalog_filepath = "/pscratch/sd/s/sajkov/data/integrated_catalog_23apr26.pq"
redshifts_filepath = "/pscratch/sd/s/sajkov/data/mock_catalog_Ch1_26.h5"

DATASET_popCosmos_full   = tables_io.read(noiseless_catalog_filepath)
with h5py.File(redshifts_filepath) as simulated_catalog:
    REDSHIFTS_popCosmos_full = simulated_catalog['sps_parameters'][:, -1]

### Number of pop-cosmos sources to use in analysis
data_cut = 11_000 #100_000 ### Note that,
                     # since the LePhare informer needs a separate sample to generate templates,
                     # this number will be inflated by {deep_field_frac}% in the final analysis.
                     # Of the number you specify, {deep_field_frac}% will indeed be reserved as `deep-field` sources
                     # and {1 - deep_field_frac}% will indeed be reserved as `WideFastDeep` sources.

### Fraction of deep field versus WFD photometry
deep_field_frac = 0.090909 #0.2

### Randomize full dataset indices, take `deep_field_frac` to be the deep field, let the rest be WFD
RANDIDX_popCosmos_full = rng.choice(np.arange(len(DATASET_popCosmos_full)), len(DATASET_popCosmos_full), replace = False)
RANDIDX_popCosmos_cut  = RANDIDX_popCosmos_full[:int(data_cut * (1 + deep_field_frac))]

deep_field_cut = int(deep_field_frac * data_cut)
IDX_popCosmos_DeepField             = RANDIDX_popCosmos_cut[:deep_field_cut]
IDX_popCosmos_DeepField_lpReference = RANDIDX_popCosmos_cut[deep_field_cut:2*deep_field_cut]
IDX_popCosmos_WideFastDeep          = RANDIDX_popCosmos_cut[2*deep_field_cut:]

### 5-sigma limiting depths ------------------
### LSST: median values for COSMOS deep field from https://usdf-maf.slac.stanford.edu/summaryStats?runId=5#Basics_Coadd%20M5
### Roman from https://github.com/jfcrenshaw/photerr/blob/a014b39729ddde3daf80be2dbe82f5a7f958882c/photerr/roman.py#L120-L126
### HSC Niji (60 min exposure) from https://sites.google.com/view/hsc-mb-survey3/filter-specification?authuser=0 
###
### Acessed June 3, 2026
### --------------------------------------------

M5_DEPTHS_DeepField = {'LSST_u'    : 27.74,
                      'LSST_g'    : 28.69,
                      'LSST_r'    : 28.88,
                      'LSST_i'    : 28.96,
                      'LSST_z'    : 28.26,
                      'LSST_y'    : 26.63,
                      'Roman_F062': 27.7,
                      'Roman_F087': 27.7,
                      'Roman_F106': 27.6,
                      'Roman_F129': 27.5,
                      'Roman_F158': 27.0,
                      'Roman_F184': 25.9,
                      'Roman_F213': 28.3,
                      'HSC_MB_00' : 26.41,
                      'HSC_MB_01' : 26.51,
                      'HSC_MB_02' : 26.45,
                      'HSC_MB_03' : 26.69,
                      'HSC_MB_04' : 26.93,
                      'HSC_MB_05' : 26.62,
                      'HSC_MB_06' : 26.26,
                      'HSC_MB_07' : 26.02,
                      'HSC_MB_08' : 26.07,
                      'HSC_MB_09' : 26.00,
                      'HSC_MB_10' : 26.06,
                      'HSC_MB_11' : 25.52,
                      'HSC_MB_12' : 25.58,
                      'HSC_MB_13' : 25.43,
                      'HSC_MB_14' : 25.15,
                      'HSC_MB_15' : 24.79}


### 5-sigma limiting depths for WideFastDeep from https://usdf-maf.slac.stanford.edu/summaryStats?runId=5#Basics_Coadd%20M5
### Column `DD:WFD CoaddM5`
M5_DEPTHS_WideFastDeep = {'LSST_u'    : 25.61,
                          'LSST_g'    : 26.90,
                          'LSST_r'    : 26.87,
                          'LSST_i'    : 26.43,
                          'LSST_z'    : 25.73,
                          'LSST_y'    : 24.79}

### Get list of bands
BANDS_DeepField = list(M5_DEPTHS_DeepField.keys())
ERR_BANDS_DeepField = [f"{key}_err" for key in BANDS_DeepField]

BANDS_WideFastDeep = list(M5_DEPTHS_WideFastDeep.keys())
ERR_BANDS_WideFastDeep = [f"{key}_err" for key in BANDS_WideFastDeep]

### Select needed bands and pick out needed sources
PHOTOMETRY_DeepField_noiseless                = DATASET_popCosmos_full[BANDS_DeepField].iloc[IDX_popCosmos_DeepField]
PHOTOMETRY_DeepField_lpReference_noiseless    = DATASET_popCosmos_full[BANDS_DeepField].iloc[IDX_popCosmos_DeepField_lpReference]
PHOTOMETRY_WideFastDeep_noiseless             = DATASET_popCosmos_full[BANDS_WideFastDeep].iloc[IDX_popCosmos_WideFastDeep]

### Same as above, for redshifts
REDSHIFTS_DeepField                = REDSHIFTS_popCosmos_full[IDX_popCosmos_DeepField]
REDSHIFTS_DeepField_lpReference    = REDSHIFTS_popCosmos_full[IDX_popCosmos_DeepField_lpReference]
REDSHIFTS_WideFastDeep             = REDSHIFTS_popCosmos_full[IDX_popCosmos_WideFastDeep]

### ### Apply noise

### Noising parameters

nYrObs     = 1 # one-year depths
nVisYr     = 1 # one visit/yr (i.e., no co-adds)
gamma      = 0.04
sigLim     = 0 # 
inputType  = 'pogson' # input pogson magnitudes (AB)
outputType = 'asinh'  # output asinh magnitudes

seed = 42

### Create: deep, multi-band, medium-band photometry for use in LePhare

print(timestamp(), "Getting noisy deep field photometry.", end = "\r")

noisyPhotometryPath_DeepField = f"{outputs_directory}/PHOTOMETRY_DeepField_noisy_{date}.pq"

getNoisyDeepFieldPhotometry = MultiSurveyErrorModel.make_stage(
    name = "getNoisyDeepFieldPhotometry",
    
    inputType  = inputType,
    outputType = outputType,

    noisy_catalog     = noisyPhotometryPath_DeepField,
    
    m5     = M5_DEPTHS_DeepField,
    bands  = BANDS_DeepField,
    nYrObs = nYrObs,
    nVisYr = nVisYr,
    gamma  = gamma,
    sigLim = sigLim,
    
    seed = seed
)

getNoisyDeepFieldPhotometry.set_data("noiseless_catalog", PHOTOMETRY_DeepField_noiseless) 
getNoisyDeepFieldPhotometry.run()
getNoisyDeepFieldPhotometry.get_handle("noisy_catalog").write()
getNoisyDeepFieldPhotometry.finalize()

### Create: the same as above, but as a template-generating reference for LePhare

print(timestamp(), "Getting noisy deep field photometry for LePhare.", end = "\r")

noisyPhotometryPath_DeepField_LePhareReference = f"{outputs_directory}/PHOTOMETRY_DeepField_noisy_lpReference_{date}.pq"

getNoisyDeepFieldPhotometry_lpReference = MultiSurveyErrorModel.make_stage(
    name = "getNoisyDeepFieldPhotometry_lpReference",
    
    inputType  = inputType,
    outputType = outputType,

    noisy_catalog     = noisyPhotometryPath_DeepField_LePhareReference,
    
    m5     = M5_DEPTHS_DeepField,
    bands  = BANDS_DeepField,
    nYrObs = nYrObs,
    nVisYr = nVisYr,
    gamma  = gamma,
    sigLim = sigLim,
    
    seed = seed
)

getNoisyDeepFieldPhotometry_lpReference.set_data("noiseless_catalog", PHOTOMETRY_DeepField_lpReference_noiseless) 
getNoisyDeepFieldPhotometry_lpReference.run()
getNoisyDeepFieldPhotometry_lpReference.get_handle("noisy_catalog").write()
getNoisyDeepFieldPhotometry_lpReference.finalize()

### Create: LSST-like photometry

print(timestamp(), "Getting noisy WideFastDeep photometry.          ", end = "\r")

noisyPhotometryPath_WideFastDeep = f"{outputs_directory}/PHOTOMETRY_WideFastDeep_noisy_{date}.pq"

getNoisyWideFastDeepPhotometry = MultiSurveyErrorModel.make_stage(
    name = "getNoisyWideFastDeepPhotometry",
    
    inputType  = inputType,
    outputType = outputType,

    noisy_catalog     = noisyPhotometryPath_WideFastDeep,
    
    m5     = M5_DEPTHS_WideFastDeep,
    bands  = BANDS_WideFastDeep,
    nYrObs = nYrObs,
    nVisYr = nVisYr,
    gamma  = gamma,
    sigLim = sigLim,
    
    seed = seed
)

getNoisyWideFastDeepPhotometry.set_data("noiseless_catalog", PHOTOMETRY_WideFastDeep_noiseless) 
getNoisyWideFastDeepPhotometry.run()
getNoisyWideFastDeepPhotometry.get_handle("noisy_catalog").write()
getNoisyWideFastDeepPhotometry.finalize()

print(timestamp(), "Finished creating datasets. Time elapsed: ", timer(start_time))

print("------------ Length of datasets ------------")
print(f"- Deep field:                    {len(PHOTOMETRY_DeepField_noiseless)}")
print(f"- Deep field, LePhare reference: {len(PHOTOMETRY_DeepField_lpReference_noiseless)}")
print(f"- WideFastDeep field:            {len(PHOTOMETRY_WideFastDeep_noiseless)}")
print("--------------------------------------------")

# ----------------------------------------------------------------------------------------------- #
# Section 2: Estimate photo-zs for deep-field sampe with LePhare                                  #
# ----------------------------------------------------------------------------------------------- #

print(timestamp(), "Informing LePhare", end = "\r")

### Inform LePhare
outputModelPath_lePhare = f"{outputs_directory}/model_lephare_{date}.pkl"

inform_lephare = LephareInformer.make_stage(
    
    name           = "inform_lephare",

    nondetect_val  = np.nan,
    model          = outputModelPath_lePhare,
    hdf5_groupname = "",
    
    bands        = BANDS_DeepField,
    err_bands    = ERR_BANDS_DeepField,
    ref_band     = "LSST_i",
    redshift_col = "redshift",
    
    zmin   = 0,
    zmax   = 6,
    nzbins = 601,
    
    **{
        "lephare.FILTER_LIST": FILTER_LIST,
        "lephare.FILTER_FILE": "filter_pipeline",
    },
)

TRAININGDATA_LePhare = getNoisyDeepFieldPhotometry_lpReference.get_handle("noisy_catalog").data.copy()
TRAININGDATA_LePhare["redshift"] = REDSHIFTS_DeepField_lpReference

inform_lephare.inform(TRAININGDATA_LePhare)
inform_lephare.get_handle("model").write()
inform_lephare.finalize()

print(timestamp(), "Informed LePhare. Time elapsed: ", timer(start_time))

print(timestamp(), "Estimating photo-zs with LePhare", end = "\r")

### Estimate with LePhare
outputEstimationPath_lePhare = f"{outputs_directory}/estimation_lephare_{date}"
estimate_lephare = LephareEstimator.make_stage(
    
    name           = "estimate_lephare",
    model          = inform_lephare.get_handle("model"),
    hdf5_groupname = "",

    bands        = BANDS_DeepField,
    err_bands    = ERR_BANDS_DeepField,
    ref_band     = "LSST_i",
    
    nondetect_val = np.nan,
    aliases = dict(input="test_data", output="lephare_estim"),
)

ESTIMATION_DATA_LePhare = getNoisyDeepFieldPhotometry.get_handle("noisy_catalog").data
lephare_estimated = estimate_lephare.estimate(ESTIMATION_DATA_LePhare)

PHOTOZS_rawLePhareOutput = lephare_estimated.read().median()
SIGMAS_photoZs_rawLePhareOutput = lephare_estimated.read().std()
PHOTOZS_DeepField_lephare = np.reshape(PHOTOZS_rawLePhareOutput, len(PHOTOZS_rawLePhareOutput))
SIGMAS_photozs_DeepField_lephare = np.reshape(SIGMAS_photoZs_rawLePhareOutput, len(SIGMAS_photoZs_rawLePhareOutput))

df_PHOTOZS_DeepField_lephare = pd.DataFrame({"photo-z": PHOTOZS_DeepField_lephare, "sigma-photo-z": SIGMAS_photozs_DeepField_lephare})
df_PHOTOZS_DeepField_lephare.to_parquet(f"{outputs_directory}/PHOTOZS_DeepField_lePhare_{date}.pq")

print(timestamp(), "Finished estimating photo-zs with LePhare. Time elapsed: ", timer(start_time))

# ----------------------------------------------------------------------------------------------- #
# Section 3: Estimate redshfits with UMAPs                                                        #
# ----------------------------------------------------------------------------------------------- #

print(timestamp(), "Preparing input data for UMAPs", end = "\r")

### Prepare input data
TRAINING_DATA_UMAP = getNoisyDeepFieldPhotometry.get_handle("noisy_catalog").data

TRAINING_PHOTOMETRY_UMAP = TRAINING_DATA_UMAP[BANDS_WideFastDeep]
TRAINING_PHOTOMERRS_UMAP = TRAINING_DATA_UMAP[ERR_BANDS_WideFastDeep]

TRAINING_REDSHIFTS_UMAP_specZs  = REDSHIFTS_DeepField
TRAINING_REDSHIFTS_UMAP_photoZs = PHOTOZS_DeepField_lephare

ESTIMATION_DATA_UMAP = getNoisyWideFastDeepPhotometry.get_handle("noisy_catalog").data

ESTIMATION_PHOTOMETRY_UMAP = ESTIMATION_DATA_UMAP[BANDS_WideFastDeep]
ESTIMATION_PHOTOMERRS_UMAP = ESTIMATION_DATA_UMAP[ERR_BANDS_WideFastDeep]

print(timestamp(), "Prepared input data for UMAPs. Time elapsed: ", timer(start_time))

print(timestamp(), "Building, and getting redshifts from, spec-z UMAP", end = "\r")

### Buld a UMAP from the training photometry, colored with spectrosopic redshifts
informedReducerPath_UMAP_wSpecZs      = f"{outputs_directory}/informedReducer_UMAP_wSpecZs_{date}.pkl"
informedEmbeddingPath_UMAP_wSpecZs    = f"{outputs_directory}/informedEmbedding_UMAP_wSpecZs_{date}.pq"
informedkNNRegressorPath_UMAP_wSpecZs = f"{outputs_directory}/informedkNNRegressor_UMAP_wSpecZs_{date}.pkl"

estimatedEmbeddingPath_UMAP_wSpecZs     = f"{outputs_directory}/estimatedEmbedding_UMAP_wSpecZs_{date}.pq"
estimatedPhotoZMediansPath_UMAP_wSpecZs = f"{outputs_directory}/estimatedPhotoZMedians_UMAP_wSpecZs_{date}.pq"
estimatedPhotoZPDFsPath_UMAP_wSpecZs    = f"{outputs_directory}/estimatedPhotoZPDFs_UMAP_wSpecZs_{date}.hdf5"

estimatePhotozsUMAP_wSpecZs = UMAPEstimator.make_stage(
    name = "estimatePhotozsUMAP_wSpecZs",

    ### Specify paths
    informed_reducer       = informedReducerPath_UMAP_wSpecZs,
    informed_embedding     = informedEmbeddingPath_UMAP_wSpecZs,
    informed_kNN_regressor = informedkNNRegressorPath_UMAP_wSpecZs,

    estimated_embedding      = estimatedEmbeddingPath_UMAP_wSpecZs,
    estimated_photoz_medians = estimatedPhotoZMediansPath_UMAP_wSpecZs,
    estimated_photoz_pdfs    = estimatedPhotoZPDFsPath_UMAP_wSpecZs,

    ### Specify UMAP parameters
    ambient_metric_umap = ambient_metric_umap,
    
    n_neighbors_umap = n_neighbors_umap,
    min_dist         = min_dist,
    
    n_neighbors_knn = n_neighbors_knn,
    metric_p_knn    = metric_p_knn,
    
    precision_gauss_kde = precision_gauss_kde,
    
    seed = seed
)

estimatePhotozsUMAP_wSpecZs.set_data("training_photometry", data = TRAINING_PHOTOMETRY_UMAP)
estimatePhotozsUMAP_wSpecZs.set_data("training_phot_error", data = TRAINING_PHOTOMERRS_UMAP)
estimatePhotozsUMAP_wSpecZs.set_data("training_redshift",   data = TRAINING_REDSHIFTS_UMAP_specZs)

estimatePhotozsUMAP_wSpecZs.UMAP_informer()

estimatePhotozsUMAP_wSpecZs.set_data("estimation_photometry", data = ESTIMATION_PHOTOMETRY_UMAP)
estimatePhotozsUMAP_wSpecZs.set_data("estimation_phot_error", data = ESTIMATION_PHOTOMERRS_UMAP)
estimatePhotozsUMAP_wSpecZs.UMAP_estimator()

estimatePhotozsUMAP_wSpecZs.get_handle("informed_reducer").write()
estimatePhotozsUMAP_wSpecZs.get_handle("informed_embedding").write()
estimatePhotozsUMAP_wSpecZs.get_handle("informed_kNN_regressor").write()
estimatePhotozsUMAP_wSpecZs.get_handle("estimated_embedding").write()
estimatePhotozsUMAP_wSpecZs.get_handle("estimated_photoz_medians").write()
estimatePhotozsUMAP_wSpecZs.get_handle("estimated_photoz_pdfs").write()

PHOTOZS_estimated_wSpecZs = estimatePhotozsUMAP_wSpecZs.get_handle("estimated_photoz_medians").data
estimatePhotozsUMAP_wSpecZs.finalize()

print(timestamp(), "Built, and got photo-zs from, spec-z UMAP. Time elapsed: ", timer(start_time))

print(timestamp(), "Building, and getting redshifts from, photoz-z UMAP", end = "\r")

### Buld a UMAP from the training photometry, colored with the photometric redshifts from LePhare
informedReducerPath_UMAP_wPhotoZs      = f"{outputs_directory}/informedReducer_UMAP_wPhotoZs_{date}.pkl"
informedEmbeddingPath_UMAP_wPhotoZs    = f"{outputs_directory}/informedEmbedding_UMAP_wPhotoZs_{date}.pq"
informedkNNRegressorPath_UMAP_wPhotoZs = f"{outputs_directory}/informedkNNRegressor_UMAP_wPhotoZs_{date}.pkl"

estimatedEmbeddingPath_UMAP_wPhotoZs     = f"{outputs_directory}/estimatedEmbedding_UMAP_wPhotoZs_{date}.pq"
estimatedPhotoZMediansPath_UMAP_wPhotoZs = f"{outputs_directory}/estimatedPhotoZMedians_UMAP_wPhotoZs_{date}.pq"
estimatedPhotoZPDFsPath_UMAP_wPhotoZs    = f"{outputs_directory}/estimatedPhotoZPDFs_UMAP_wPhotoZs_{date}.hdf5"

estimatePhotozsUMAP_wPhotoZs = UMAPEstimator.make_stage(
    name = "estimatePhotozsUMAP_wPhotoZs",

    ### Specify paths
    informed_reducer       = informedReducerPath_UMAP_wPhotoZs,
    informed_embedding     = informedEmbeddingPath_UMAP_wPhotoZs,
    informed_kNN_regressor = informedkNNRegressorPath_UMAP_wPhotoZs,

    estimated_embedding      = estimatedEmbeddingPath_UMAP_wPhotoZs,
    estimated_photoz_medians = estimatedPhotoZMediansPath_UMAP_wPhotoZs,
    estimated_photoz_pdfs    = estimatedPhotoZPDFsPath_UMAP_wPhotoZs,

    ### Specify UMAP parameters
    ambient_metric_umap = ambient_metric_umap,
    
    n_neighbors_umap = n_neighbors_umap,
    min_dist         = min_dist,
    
    n_neighbors_knn = n_neighbors_knn,
    metric_p_knn    = metric_p_knn,
    
    precision_gauss_kde = precision_gauss_kde,
    
    seed = seed
)

estimatePhotozsUMAP_wPhotoZs.set_data("training_photometry", data = TRAINING_PHOTOMETRY_UMAP)
estimatePhotozsUMAP_wPhotoZs.set_data("training_phot_error", data = TRAINING_PHOTOMERRS_UMAP)
estimatePhotozsUMAP_wPhotoZs.set_data("training_redshift",   data = TRAINING_REDSHIFTS_UMAP_photoZs)

estimatePhotozsUMAP_wPhotoZs.UMAP_informer()

estimatePhotozsUMAP_wPhotoZs.set_data("estimation_photometry", data = ESTIMATION_PHOTOMETRY_UMAP)
estimatePhotozsUMAP_wPhotoZs.set_data("estimation_phot_error", data = ESTIMATION_PHOTOMERRS_UMAP)
estimatePhotozsUMAP_wPhotoZs.UMAP_estimator()

estimatePhotozsUMAP_wPhotoZs.get_handle("informed_reducer").write()
estimatePhotozsUMAP_wPhotoZs.get_handle("informed_embedding").write()
estimatePhotozsUMAP_wPhotoZs.get_handle("informed_kNN_regressor").write()
estimatePhotozsUMAP_wPhotoZs.get_handle("estimated_embedding").write()
estimatePhotozsUMAP_wPhotoZs.get_handle("estimated_photoz_medians").write()
estimatePhotozsUMAP_wPhotoZs.get_handle("estimated_photoz_pdfs").write()

PHOTOZS_estimated_wPhotoZs = estimatePhotozsUMAP_wPhotoZs.get_handle("estimated_photoz_medians").data
estimatePhotozsUMAP_wPhotoZs.finalize()

print(timestamp(), "Built, and got photo-zs from, photo-z UMAP. Time elapsed: ", timer(start_time))

print("Pipeline finished.")