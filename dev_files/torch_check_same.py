import numpy as np

def rel_L2(pred, act):
    pred = np.asarray(pred)
    act  = np.asarray(act)

    if pred.size == 1:
        num = abs((pred-act).item())
        den = abs(pred.item())
    else:
        num = np.linalg.norm(pred-act, ord=2)
        den = np.linalg.norm(pred)
    if den == 0:
        return num #Absolute Error
    return num/den

new_file = 'Results/torch_output.npz'
saved_file = 'Results/torch_output_saved.npz'

new_results = np.load(new_file)
saved_results = np.load(saved_file)

for key in saved_results:
    print("Checking Key:", key)
    if np.allclose(saved_results[key], new_results[key]):
        print("Results Correct")
    else:
        print("ERROR, Misaligned Result")
        print("L2 Diff: ", rel_L2(new_results[key], saved_results[key]))
        # print("Saved:", saved_results[key])
        # print("New", new_results[key])
        