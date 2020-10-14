import numpy as np


class GetData:
    """A class to shuffle data and generate batches for each iteration"""

    def __init__(self, data, target, batch_size, shuffle_data=True):
        self.shuffle_data = shuffle_data
        if shuffle_data:
            data_idx = np.random.permutation(len(data))
        else:
            data_idx = range(len(data))

        self.data = data[data_idx]
        self.target = target[data_idx]
        self.batch_size = batch_size
        self.n_batches = int(np.ceil(data.shape[0] / batch_size))
        self.iter = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter < self.n_batches:
            batch = self.data[
                    (
                            self.iter * self.batch_size):(
                            self.iter + 1 * self.batch_size)]
            batch_target = self.target[
                           (
                                   self.iter * self.batch_size):(
                                   self.iter + 1 * self.batch_size)]
            self.iter += 1
            return batch, batch_target
        else:
            if self.shuffle_data:
                data_idx = np.random.permutation(len(self.data))
            else:
                data_idx = range(len(self.data))
            self.data = self.data[data_idx]
            self.target = self.target[data_idx]
