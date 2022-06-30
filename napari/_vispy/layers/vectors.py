import logging

import numpy as np

from ..visuals.vectors import VectorsVisual
from .base import VispyBaseLayer
from napari.layers.base.base import LayerSliceResponse

LOGGER = logging.getLogger("napari._vispy.layers.vectors")


class VispyVectorsLayer(VispyBaseLayer):
    def __init__(self, layer):
        node = VectorsVisual()
        super().__init__(layer, node)

        self.layer.events.edge_color.connect(self._on_data_change)

        self.reset()
        # self._on_data_change()

    def _set_slice(self, response: LayerSliceResponse) -> None:
        """This method replaces the old on_data_change
        
        Response data needs face_color
        """
        LOGGER.debug('VispyPointsLayer._set_slice : %s', response.request)

        # this logic need to move to the get_slice layer class
        # if (
        #     len(self.layer._view_vertices) == 0
        #     or len(self.layer._view_faces) == 0
        # ):
        #     vertices = np.zeros((3, self.layer._ndisplay))
        #     faces = np.array([[0, 1, 2]])
        #     face_color = np.array([[0, 0, 0, 0]])
        # else:
        #     vertices = self.layer._view_vertices[:, ::-1]
        #     faces = self.layer._view_faces
        #     face_color = self.layer._view_face_color

        # if self.layer._ndisplay == 3 and self.layer.ndim == 2:
        #     vertices = np.pad(vertices, ((0, 0), (0, 1)), mode='constant')


        # the commented lines below were already commented in main
        # self.node.set_data(
        #     vertices=vertices, faces=faces, color=self.layer.current_edge_color
        # )
        self.node.set_data(
            vertices=response.data.vertices,
            faces=response.data.faces,
            face_colors=response.data.face_color,
        )

        self.node.update()
        # Call to update order of translation values with new dims:
        self._on_matrix_change()

    def _on_data_change(self):
        """this is replaced by set_slice, all the layer data gets moved out"""
        raise Exception("Deprecated method; should not be called")
        if (
            len(self.layer._view_vertices) == 0
            or len(self.layer._view_faces) == 0
        ):
            vertices = np.zeros((3, self.layer._ndisplay))
            faces = np.array([[0, 1, 2]])
            face_color = np.array([[0, 0, 0, 0]])
        else:
            vertices = self.layer._view_vertices[:, ::-1]
            faces = self.layer._view_faces
            face_color = self.layer._view_face_color

        if self.layer._ndisplay == 3 and self.layer.ndim == 2:
            vertices = np.pad(vertices, ((0, 0), (0, 1)), mode='constant')

        # self.node.set_data(
        #     vertices=vertices, faces=faces, color=self.layer.current_edge_color
        # )
        self.node.set_data(
            vertices=vertices,
            faces=faces,
            face_colors=face_color,
        )

        self.node.update()
        # Call to update order of translation values with new dims:
        self._on_matrix_change()

