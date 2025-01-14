import keras
import numpy as np
import h5py


class ECGPyDataset(keras.utils.PyDataset):

    #x_set: list of ECG reading file numbers
    #y_set: list of correct classes
    def __init__(self, x_set, y_set, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    # Return number of batches
    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    # Return x, y for batch number idx
    def __getitem__(self, idx):
        low = idx * self.batch_size
        high = min(low + self.batch_size, len(self.x))

        batch_x = np.array([self._read_ecg(file) for file in self.x[low:high]])
        print("low", low, "high", high)
        batch_y = np.array(self.y[low:high])
        print("x shape:", batch_x.shape)
        print("y shape:", batch_y.shape)

        return [batch_x, batch_y]

    #Extract ECG tracings from their h5 file to a numpy array
    def _read_ecg(self, n):
        with h5py.File(f'data/records/A{n:05d}.h5', 'r') as f:
            signal = f['ecg'][()]
            #Take the first 10s of each reading
            return signal[:,:5000]
