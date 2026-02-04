from scipy.io import readsav

file_path = 'kn1d/jh_bscoef.dat'
sav_data = readsav(file_path)

# print("LogR_BSCoef", sav_data['logr_bscoef'])
# input()
print("LogS_BSCoef", sav_data['logs_bscoef'])