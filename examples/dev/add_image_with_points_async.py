# A simple driving example to test async slicing.

from skimage import data
import numpy as np
import napari

image_data = np.random.rand(100, 1024, 1024)

# 1 million points for 10000 points per slice, as in points example
np.random.seed(0)
n = 1_000_000
points_data = image_data.shape * np.random.rand(n, 3)

viewer = napari.Viewer()
viewer.add_image(image_data)
viewer.add_points(points_data)

if __name__ == '__main__':
    napari.run()
