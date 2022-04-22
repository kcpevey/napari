# A simple driving example to test async slicing.

import numpy as np
import napari

image_data = np.random.rand(100, 1024, 1024)

viewer = napari.view_image(image_data)

if __name__ == '__main__':
    napari.run()
