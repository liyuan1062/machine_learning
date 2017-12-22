# coding=utf-8

import numpy as np
from keras.preprocessing.image import ImageDataGenerator

class CSVData(object):
    def __init__(self, file_path, test=False, data_type=np.float32, delimiter=','):
        self.file_path = file_path
        self.data_type = data_type
        self.delimiter = delimiter
        self.all_data = None
        self.data = None
        self.label = None
        self.test = test
        self.loadData()

    def loadData(self):
        self.all_data = np.loadtxt(self.file_path, dtype=np.str, delimiter=self.delimiter)
        if not self.test:
            self.data = self.all_data[1:, 1:].astype(self.data_type)
            self.label = self.all_data[1:, 0].astype(self.data_type)
        else:
            self.data = self.all_data[1:, :].astype(self.data_type)
            self.label = None
        self.normalized_data = np.multiply(self.data, 1.0/self.data.max())

    def shape(self):
        return self.data.shape

    def get_label_count(self):
        return np.unique(self.label).size

    def get_one_hot_labels(self):
        label_count = np.unique(self.label).size
        labels_one_hot = np.zeros((self.label.size, label_count))
        index_offset = np.arange(self.label.size) * label_count
        labels_one_hot.flat[index_offset + self.label.astype(int).ravel()] = 1
        return labels_one_hot

    def data_generator(self):
        datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=10,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=False,
            vertical_flip=False)
        # datagen.fit(np.reshape(self.data), [-1,28,28,1])
        return datagen
