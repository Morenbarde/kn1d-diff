import numpy as np

new_file = 'Results/output.npz'
saved_file = 'Results/output_saved.npz'

new_results = np.load(new_file)
saved_results = np.load(saved_file)

for key in saved_results:
    print("Checking Key:", key)
    if np.allclose(saved_results[key], new_results[key]):
        print("Results Correct")
    else:
        print("ERROR, Misaligned Result")
        print("Saved:", saved_results[key])
        print("New", new_results[key])
        break