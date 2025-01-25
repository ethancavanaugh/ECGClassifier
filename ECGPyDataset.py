import pandas as pd
import numpy as np
import keras
import h5py
import math


class ECGPyDataset(keras.utils.PyDataset):

    #x_set: list of ECG reading file numbers
    #y_set: list of correct classes
    def __init__(self, df, batch_size, **kwargs):
        super().__init__(**kwargs)

        self.x = df.ECG_ID
        self.y = df.drop('ECG_ID', axis=1)
        self.batch_size = batch_size

    # Return number of batches
    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    # Return x, y for batch number idx
    def __getitem__(self, idx):
        low = idx * self.batch_size
        high = min(low + self.batch_size, len(self.x))

        batch_x = np.array([self._read_ecg(file) for file in self.x[low:high]])
        batch_x = np.swapaxes(batch_x, 1, 2)
        batch_y = np.array(self.y[low:high])
        #print("x shape:", batch_x.shape)
        #print("y shape:", batch_y.shape)

        return batch_x, batch_y

    #Extract ECG tracings from their h5 file to a numpy array
    def _read_ecg(self, ecg_id):
        with h5py.File(f'data/records_norm/{ecg_id:s}.h5', 'r') as f:
            signal = f['ecg'][()]
            #Take the first 10s of each reading
            return signal[:,:4096]
