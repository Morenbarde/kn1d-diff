from scipy.io import readsav
import numpy as np

file_path = 'dev_files/jh_bscoef.dat'
print("Loading jh_bscoef_dat")
sav_data = readsav(file_path)

np.savez("jh_bscoef",
    DKnot = sav_data["dknot"],
    TKnot = sav_data["tknot"],
    order = sav_data["order"],
    LogR_BSCoef = sav_data["logr_bscoef"],
    LogS_BSCoef = sav_data["logs_bscoef"],
    LogAlpha_BSCoef = sav_data["logalpha_bscoef"],
    A_Lyman = sav_data["a_lyman"],
    A_Balmer = sav_data["a_balmer"])

print("Converted to jh_bscoef.npz")

