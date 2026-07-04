import os, shutil, glob
import numpy as np

path_filters_nm = "/pscratch/sd/s/sajkov/analysis_pipeline/filters"
os.makedirs(f"{path_filters_nm}/filters_nm")

for file in glob.glob(f"{path_filters_nm}/*.dat"):
    shutil.copy(file, f"{path_filters_nm}/filters_nm/{file.split("/")[-1]}")
    filter_file_content = np.loadtxt(file)
    filter_in_angstrom = filter_file_content
    filter_in_angstrom[:, 0] *= 10
    np.savetxt(file, filter_in_angstrom)