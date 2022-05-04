# A simple driving example to test async slicing.

import numpy as np
import napari
from skimage.transform import pyramid_gaussian

hires_data = np.random.rand(1000, 256, 256)
multiscale = list(
    pyramid_gaussian(hires_data, downscale=2, max_layer=4, multichannel=True)
)

viewer = napari.view_image(multiscale, multiscale=True)

if __name__ == '__main__':
    napari.run()
