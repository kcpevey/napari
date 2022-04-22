# A simple driving example to test async slicing.

import numpy as np
import napari

image_data = np.random.rand(1000, 256, 256)

viewer = napari.view_image(image_data)

if __name__ == '__main__':
    napari.run()
