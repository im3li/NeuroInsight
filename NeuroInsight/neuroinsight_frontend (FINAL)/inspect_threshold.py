import pickle
with open("models/optimal_threshold.pkl", "rb") as f:
    data = pickle.load(f)
print(type(data), data)