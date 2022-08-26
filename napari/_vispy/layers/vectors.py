import logging

import numpy as np

from ..visuals.vectors import VectorsVisual
from .base import VispyBaseLayer
from ...layers.base.base import _LayerSliceResponse

LOGGER = logging.getLogger("napari._vispy.layers.vectors")


class VispyVectorsLayer(VispyBaseLayer):
    def __init__(self, layer):
        node = VectorsVisual()
        super().__init__(layer, node)

        self.layer.events.edge_color.connect(self._on_data_change)

        self.reset()
        # self._on_data_change()

    def _set_slice(self, response: _LayerSliceResponse) -> None:
        """This method replaces the old on_data_change"""
        LOGGER.debug('VispyVectorsLayer._set_slice : %s', response.request)

        # self.node.set_data(
        #     vertices=vertices, faces=faces, color=self.layer.current_edge_color
        # )
        self.node.set_data(
            vertices=response.view_vertices,
            faces=response.view_faces,
            face_colors=response.view_face_color,
        )

        self.node.update()
        # Call to update order of translation values with new dims:
        self._on_matrix_change()

    def _on_data_change(self):
        """this is replaced by set_slice, all the layer data gets moved out"""
        raise Exception("Deprecated method; should not be called")
