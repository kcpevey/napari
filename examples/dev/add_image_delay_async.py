# A simple driving example to test async slicing.

import numpy as np
import napari
from time import sleep


class DelayedArray:

    def __init__(self, array, *, delay_s: float = 0.5):
        self.array = array
        self.delay_s = delay_s

    def __getitem__(self, key):
        sleep(self.delay_s)
        return self.array.__getitem__(key)

    @property
    def dtype(self):
        return self.array.dtype

    @property
    def shape(self):
        return self.array.shape

image_data = DelayedArray(np.random.rand(1000, 256, 256))

# Explicitly set multiscale because other guess_multiscale
# reads each slice with the delay.
viewer = napari.view_image(image_data, multiscale=False)

if __name__ == '__main__':
    napari.run()
