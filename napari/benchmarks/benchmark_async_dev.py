# Experimenting with tests to test future async layers
#
#

import time

import dask.array as da
import numpy as np
import zarr
from qtpy.QtWidgets import QApplication

import napari
from napari.layers import Image


class AsyncImage2DSuite:
    chunksize = [256, 512, 1024, 2048]
    latency = [0.05 * i for i in range(0, 4)]

    params = (latency, chunksize)

    def setup(self, latency, chunksize):
        @da.as_gufunc(signature="()->()")
        def slow_data(x):
            time.sleep(latency)
            return x

        self.zarr = zarr.zeros(
            (64, 2048, 2048), chunks=(1, chunksize, chunksize), dtype='uint8'
        )
        self.zarr[:, :, :] = np.random.randint(0, 255, size=self.zarr.shape)
        self.data = slow_data(da.from_zarr(self.zarr))

        self.layer = Image(self.data)

    def time_create_layer(self, *args):
        """Time to create an image layer."""
        Image(self.data)

    def time_set_view_slice(self, *args):
        """Time to set view slice."""
        self.layer._set_view_slice()

    def time_refresh(self, *args):
        """Time to refresh view."""
        self.layer.refresh()


class QtViewerAsyncImage2DSuite:
    chunksize = [256, 512, 1024, 2048]
    latency = [0.05 * i for i in range(0, 4)]
    params = (latency, chunksize)

    def setup(self, latency, chunksize):
        _ = QApplication.instance() or QApplication([])

        @da.as_gufunc(signature="()->()")
        def slow_data(x):
            time.sleep(latency)
            return x

        self.zarr = zarr.zeros(
            (32, 2048, 2048), chunks=(1, chunksize, chunksize), dtype='uint8'
        )
        self.zarr[:, :, :] = np.random.randint(0, 255, size=self.zarr.shape)
        self.data = slow_data(da.from_zarr(self.zarr))

        self.viewer = napari.Viewer()
        self.viewer.add_image(self.data)
        # self.viewer = napari.view_image(self.data)

    def time_z_scroll(self, *args):
        for z in range(self.data.shape[0]):
            self.viewer.dims.set_current_step(0, z)


class QtViewerAsyncPointsSuite:
    n_points = [2**i for i in range(12, 18)]
    params = n_points

    def setup(self, n_points):
        _ = QApplication.instance() or QApplication([])

        np.random.seed(0)
        self.viewer = napari.Viewer()
        # Fake image layer to set bounds. Is this really needed?
        self.empty_image = np.zeros((512, 512, 512), dtype="uint8")
        self.viewer.add_image(self.empty_image)
        self.point_data = np.random.randint(512, size=(n_points, 3))
        self.viewer.add_points(self.point_data)

    def time_z_scroll(self, *args):
        for z in range(self.empty_image.shape[0]):
            self.viewer.dims.set_current_step(0, z)


class QtViewerAsyncPointsAndImage2DSuite:
    n_points = [2**i for i in range(12, 18, 2)]
    chunksize = [256, 512, 1024]
    latency = [0.05 * i for i in range(0, 3)]
    params = (n_points, latency, chunksize)

    def setup(self, n_points, latency, chunksize):
        _ = QApplication.instance() or QApplication([])

        np.random.seed(0)

        @da.as_gufunc(signature="()->()")
        def slow_data(x):
            time.sleep(latency)
            return x

        self.zarr = zarr.zeros(
            (32, 2048, 2048), chunks=(1, chunksize, chunksize), dtype='uint8'
        )
        self.zarr[:, :, :] = np.random.randint(0, 255, size=self.zarr.shape)
        self.image_data = slow_data(da.from_zarr(self.zarr))

        self.viewer = napari.Viewer()
        self.viewer.add_image(self.image_data)
        self.point_data = np.random.randint(512, size=(n_points, 3))
        self.viewer.add_points(self.point_data)

    def time_z_scroll(self, *args):
        for z in range(self.image_data.shape[0]):
            self.viewer.dims.set_current_step(0, z)
