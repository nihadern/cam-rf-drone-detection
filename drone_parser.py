import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler


def rf_data_parse(filenames: list, normalize: bool = False) -> list:
    from sklearn.preprocessing import MinMaxScaler
    data = []
    for file in filenames:
        with open(file, 'r') as f:
            segment = f.read().split(',')
        segment = list(map(float, segment))
        segment = np.array(segment)
        if normalize:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler = scaler.fit(segment.reshape(-1, 1))
            segment = scaler.transform(segment.reshape(-1, 1))
        if len(segment) != 10000000:
            print("{} had segment with length {}".format(file, len(segment)))
        data.append(segment)
    return np.array(data)


curr_dir = os.getcwd()
join = os.path.join

drone_dir_H = ["RF Data_10011_H", "RF Data_10111_H"]
drone_dir_L = ["RF Data_10011_L", "RF Data_10111_L"]
drone_dir_H = list(map(lambda x: join("data", x), drone_dir_H))
drone_dir_L = list(map(lambda x: join("data", x), drone_dir_L))
drone_dataset = []
drone_dataset_norm = []


for i, folder in enumerate(drone_dir_L):
    path_L = join(curr_dir, folder)
    path_H = join(curr_dir, drone_dir_H[i])
    files_L = os.listdir(folder)
    files_H = os.listdir(drone_dir_H[i])
    for j, file_L in enumerate(files_L):
        file_L = join(path_L, file_L)
        file_H = join(path_H, files_H[j])
        data = rf_data_parse([file_L, file_H])
        data_norm = rf_data_parse([file_L, file_H], normalize=True)
        drone_dataset.append(data)
        drone_dataset_norm.append(data_norm)
        print("Finished processing {} and {}".format(file_L, file_H))

np.save('drone_rf_LH', np.array(drone_dataset))
np.save('drone_rf_LH_normalized', np.array(drone_dataset_norm))
