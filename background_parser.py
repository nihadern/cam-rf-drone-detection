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

bg_dir_H = ["RF Data_00000_H", "FR Data_00000_H2"]
bg_dir_L = ["RF Data_00000_L1", "RF Data_00000_L2"]
bg_dataset = []
bg_dataset_norm = []


for i, folder in enumerate(bg_dir_L):
    path_L = join(curr_dir, folder)
    path_H = join(curr_dir, bg_dir_H[i])
    files_L = os.listdir(folder)
    files_H = os.listdir(bg_dir_H[i])
    for i, file_L in enumerate(files_L):
        file_L = join(path_L, file_L)
        file_H = join(path_H, files_H[i])
        data = rf_data_parse([file_L, file_H])
        data_norm = rf_data_parse([file_L, file_H], normalize=True)
        bg_dataset.append(data)
        bg_dataset_norm.append(data_norm)
        print("Finished processing {} and {}".format(file_L, file_H))

np.save('background_rf_LH', np.array(bg_dataset))
np.save('background_rf_LH_normalized', np.array(bg_dataset_norm))
