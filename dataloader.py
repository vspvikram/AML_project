import os
import numpy as np
import tensorflow as tf


class DataLoader(tf.keras.utils.Sequence):
    def __init__(self, dataset, batch_size=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indices = np.arange(len(dataset))
        
        
    def __getitem__(self, i):
        ind_start = i*self.batch_size
        ind_end = (i+1)*self.batch_size
        
        output = [self.dataset[self.indices[j]] for j in range(ind_start, ind_end)]
        output = [np.stack(item, axis=0) for item in zip(*output)]
        print(len(output))
        
        return output
    
    def __len__(self):
        return len(self.dataset)//self.batch_size