"""Image class.
"""
from __future__ import annotations

import logging
import types
import warnings
from math import ceil
from typing import List, Sequence, Tuple, Union

import numpy as np
from scipy import ndimage as ndi

from napari.layers.base.base import LayerSliceRequest, LayerSliceResponse
from napari.utils.transforms.transforms import Affine

from ...utils import config
from ...utils._dtype import get_dtype_limits, normalize_dtype
from ...utils.colormaps import AVAILABLE_COLORMAPS
from ...utils.events import Event
from ...utils.naming import magic_name
from ...utils.translations import trans
from .._data_protocols import LayerDataProtocol
from .._multiscale_data import MultiScaleData
from ..base import Layer, no_op
from ..intensity_mixin import IntensityVisualizationMixin
from ..utils.layer_utils import calc_data_range
from ..utils.plane import SlicingPlane
from ._image_constants import (
    ImageRendering,
    Interpolation,
    Interpolation3D,
    Mode,
    VolumeDepiction,
)
from ._image_mouse_bindings import (
    move_plane_along_normal as plane_drag_callback,
)
from ._image_mouse_bindings import (
    set_plane_position as plane_double_click_callback,
)
from ._image_utils import guess_multiscale, guess_rgb

LOGGER = logging.getLogger("napari.layers.image")


# It is important to contain at least one abstractmethod to properly exclude this class
# in creating NAMES set inside of napari.layers.__init__
# Mixin must come before Layer
class _ImageBase(IntensityVisualizationMixin, Layer):
    """Image layer.

    Parameters
    ----------
    data : array or list of array
        Image data. Can be N >= 2 dimensional. If the last dimension has length
        3 or 4 can be interpreted as RGB or RGBA if rgb is `True`. If a
        list and arrays are decreasing in shape then the data is treated as
        a multiscale image. Please note multiscale rendering is only
        supported in 2D. In 3D, only the lowest resolution scale is
        displayed.
    rgb : bool
        Whether the image is rgb RGB or RGBA. If not specified by user and
        the last dimension of the data has length 3 or 4 it will be set as
        `True`. If `False` the image is interpreted as a luminance image.
    colormap : str, napari.utils.Colormap, tuple, dict
        Colormap to use for luminance images. If a string must be the name
        of a supported colormap from vispy or matplotlib. If a tuple the
        first value must be a string to assign as a name to a colormap and
        the second item must be a Colormap. If a dict the key must be a
        string to assign as a name to a colormap and the value must be a
        Colormap.
    contrast_limits : list (2,)
        Color limits to be used for determining the colormap bounds for
        luminance images. If not passed is calculated as the min and max of
        the image.
    gamma : float
        Gamma correction for determining colormap linearity. Defaults to 1.
    interpolation : str
        Interpolation mode used by vispy. Must be one of our supported
        modes.
    rendering : str
        Rendering mode used by vispy. Must be one of our supported
        modes.
    depiction : str
        3D Depiction mode. Must be one of {'volume', 'plane'}.
        The default value is 'volume'.
    iso_threshold : float
        Threshold for isosurface.
    attenuation : float
        Attenuation rate for attenuated maximum intensity projection.
    name : str
        Name of the layer.
    metadata : dict
        Layer metadata.
    scale : tuple of float
        Scale factors for the layer.
    translate : tuple of float
        Translation values for the layer.
    rotate : float, 3-tuple of float, or n-D array.
        If a float convert into a 2D rotation matrix using that value as an
        angle. If 3-tuple convert into a 3D rotation matrix, using a yaw,
        pitch, roll convention. Otherwise assume an nD rotation. Angles are
        assumed to be in degrees. They can be converted from radians with
        np.degrees if needed.
    shear : 1-D array or n-D array
        Either a vector of upper triangular values, or an nD shear matrix with
        ones along the main diagonal.
    affine : n-D array or napari.utils.transforms.Affine
        (N+1, N+1) affine transformation matrix in homogeneous coordinates.
        The first (N, N) entries correspond to a linear transform and
        the final column is a length N translation vector and a 1 or a napari
        `Affine` transform object. Applied as an extra transform on top of the
        provided scale, rotate, and shear values.
    opacity : float
        Opacity of the layer visual, between 0.0 and 1.0.
    blending : str
        One of a list of preset blending modes that determines how RGB and
        alpha values of the layer visual get mixed. Allowed values are
        {'opaque', 'translucent', and 'additive'}.
    visible : bool
        Whether the layer visual is currently being displayed.
    multiscale : bool
        Whether the data is a multiscale image or not. Multiscale data is
        represented by a list of array like image data. If not specified by
        the user and if the data is a list of arrays that decrease in shape
        then it will be taken to be multiscale. The first image in the list
        should be the largest. Please note multiscale rendering is only
        supported in 2D. In 3D, only the lowest resolution scale is
        displayed.
    cache : bool
        Whether slices of out-of-core datasets should be cached upon retrieval.
        Currently, this only applies to dask arrays.
    plane : dict or SlicingPlane
        Properties defining plane rendering in 3D. Properties are defined in
        data coordinates. Valid dictionary keys are
        {'position', 'normal', 'thickness', and 'enabled'}.
    experimental_clipping_planes : list of dicts, list of ClippingPlane, or ClippingPlaneList
        Each dict defines a clipping plane in 3D in data coordinates.
        Valid dictionary keys are {'position', 'normal', and 'enabled'}.
        Values on the negative side of the normal are discarded if the plane is enabled.

    Attributes
    ----------
    data : array or list of array
        Image data. Can be N dimensional. If the last dimension has length
        3 or 4 can be interpreted as RGB or RGBA if rgb is `True`. If a list
        and arrays are decreasing in shape then the data is treated as a
        multiscale image. Please note multiscale rendering is only
        supported in 2D. In 3D, only the lowest resolution scale is
        displayed.
    metadata : dict
        Image metadata.
    rgb : bool
        Whether the image is rgb RGB or RGBA if rgb. If not
        specified by user and the last dimension of the data has length 3 or 4
        it will be set as `True`. If `False` the image is interpreted as a
        luminance image.
    multiscale : bool
        Whether the data is a multiscale image or not. Multiscale data is
        represented by a list of array like image data. The first image in the
        list should be the largest. Please note multiscale rendering is only
        supported in 2D. In 3D, only the lowest resolution scale is
        displayed.
    mode : str
        Interactive mode. The normal, default mode is PAN_ZOOM, which
        allows for normal interactivity with the canvas.

        In TRANSFORM mode the image can be transformed interactively.
    colormap : 2-tuple of str, napari.utils.Colormap
        The first is the name of the current colormap, and the second value is
        the colormap. Colormaps are used for luminance images, if the image is
        rgb the colormap is ignored.
    colormaps : tuple of str
        Names of the available colormaps.
    contrast_limits : list (2,) of float
        Color limits to be used for determining the colormap bounds for
        luminance images. If the image is rgb the contrast_limits is ignored.
    contrast_limits_range : list (2,) of float
        Range for the color limits for luminance images. If the image is
        rgb the contrast_limits_range is ignored.
    gamma : float
        Gamma correction for determining colormap linearity.
    interpolation : str
        Interpolation mode used by vispy. Must be one of our supported
        modes.
    rendering : str
        Rendering mode used by vispy. Must be one of our supported
        modes.
    depiction : str
        3D Depiction mode used by vispy. Must be one of our supported modes.
    iso_threshold : float
        Threshold for isosurface.
    attenuation : float
        Attenuation rate for attenuated maximum intensity projection.
    plane : SlicingPlane or dict
        Properties defining plane rendering in 3D. Valid dictionary keys are
        {'position', 'normal', 'thickness'}.
    experimental_clipping_planes : ClippingPlaneList
        Clipping planes defined in data coordinates, used to clip the volume.

    Notes
    -----
    _colorbar : array
        Colorbar for current colormap.
    """

    _colormaps = AVAILABLE_COLORMAPS

    def __init__(
        self,
        data,
        *,
        rgb=None,
        colormap='gray',
        contrast_limits=None,
        gamma=1,
        interpolation='nearest',
        rendering='mip',
        iso_threshold=0.5,
        attenuation=0.05,
        name=None,
        metadata=None,
        scale=None,
        translate=None,
        rotate=None,
        shear=None,
        affine=None,
        opacity=1,
        blending='translucent',
        visible=True,
        multiscale=None,
        cache=True,
        depiction='volume',
        plane=None,
        experimental_clipping_planes=None,
    ):
        if name is None and data is not None:
            name = magic_name(data)

        if isinstance(data, types.GeneratorType):
            data = list(data)

        if getattr(data, 'ndim', 2) < 2:
            raise ValueError(
                trans._('Image data must have at least 2 dimensions.')
            )

        # Determine if data is a multiscale
        self._data_raw = data
        if multiscale is None:
            multiscale, data = guess_multiscale(data)
        elif multiscale and not isinstance(data, MultiScaleData):
            data = MultiScaleData(data)

        # Determine if rgb
        rgb_guess = guess_rgb(data.shape)
        if rgb and not rgb_guess:
            raise ValueError(
                trans._(
                    "'rgb' was set to True but data does not have suitable dimensions."
                )
            )
        elif rgb is None:
            rgb = rgb_guess

        # Determine dimensionality of the data
        ndim = len(data.shape)
        if rgb:
            ndim -= 1

        super().__init__(
            data,
            ndim,
            name=name,
            metadata=metadata,
            scale=scale,
            translate=translate,
            rotate=rotate,
            shear=shear,
            affine=affine,
            opacity=opacity,
            blending=blending,
            visible=visible,
            multiscale=multiscale,
            cache=cache,
            experimental_clipping_planes=experimental_clipping_planes,
        )

        self.events.add(
            mode=Event,
            interpolation=Event,
            rendering=Event,
            depiction=Event,
            iso_threshold=Event,
            attenuation=Event,
        )

        self._array_like = True

        # Set data
        self.rgb = rgb
        self._data = data
        if self.multiscale:
            self._data_level = len(self.data) - 1
            # Determine which level of the multiscale to use for the thumbnail.
            # Pick the smallest level with at least one axis >= 64. This is
            # done to prevent the thumbnail from being from one of the very
            # low resolution layers and therefore being very blurred.
            big_enough_levels = [
                np.any(np.greater_equal(p.shape, 64)) for p in data
            ]
            if np.any(big_enough_levels):
                self._thumbnail_level = np.where(big_enough_levels)[0][-1]
            else:
                self._thumbnail_level = 0
        else:
            self._data_level = 0
            self._thumbnail_level = 0
        displayed_axes = self._displayed_axes
        self.corner_pixels[1][displayed_axes] = self.level_shapes[
            self._data_level
        ][displayed_axes]

        # Set contrast limits, colormaps and plane parameters
        self._gamma = gamma
        self._iso_threshold = iso_threshold
        self._attenuation = attenuation
        self._plane = SlicingPlane(thickness=1, enabled=False, draggable=True)
        self._mode = Mode.PAN_ZOOM
        # Whether to calculate clims on the next set_view_slice
        self._should_calc_clims = False
        if contrast_limits is None:
            if not isinstance(data, np.ndarray):
                dtype = normalize_dtype(getattr(data, 'dtype', None))
                if np.issubdtype(dtype, np.integer):
                    self.contrast_limits_range = get_dtype_limits(dtype)
                else:
                    self.contrast_limits_range = (0, 1)
                self._should_calc_clims = dtype != np.uint8
            else:
                self.contrast_limits_range = self._calc_data_range()
        else:
            self.contrast_limits_range = contrast_limits
        self._contrast_limits = tuple(self.contrast_limits_range)
        # using self.colormap = colormap uses the setter in *derived* classes,
        # where the intention here is to use the base setter, so we use the
        # _set_colormap method. This is important for Labels layers, because
        # we don't want to use get_color before set_view_slice has been
        # triggered (self._update_dims(), below).
        self._set_colormap(colormap)
        self.contrast_limits = self._contrast_limits
        self._interpolation = {
            2: Interpolation.NEAREST,
            3: (
                Interpolation3D.NEAREST
                if self.__class__.__name__ == 'Labels'
                else Interpolation3D.LINEAR
            ),
        }
        self.interpolation = interpolation
        self.rendering = rendering
        self.depiction = depiction
        if plane is not None:
            self.plane = plane

        # Trigger generation of view slice and thumbnail
        self._update_dims()

    def _get_order(self):
        """Return the order of the displayed dimensions."""
        if self.rgb:
            # if rgb need to keep the final axis fixed during the
            # transpose. The index of the final axis depends on how many
            # axes are displayed.
            return self._dims_displayed_order + (
                max(self._dims_displayed_order) + 1,
            )
        else:
            return self._dims_displayed_order

    def _calc_data_range(self, mode='data'):
        if mode == 'data':
            input_data = self.data[-1] if self.multiscale else self.data
        elif mode == 'slice':
            data = self._slice.image.view  # ugh
            input_data = data[-1] if self.multiscale else data
        else:
            raise ValueError(
                trans._(
                    "mode must be either 'data' or 'slice', got {mode!r}",
                    deferred=True,
                    mode=mode,
                )
            )
        return calc_data_range(input_data, rgb=self.rgb)

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def data_raw(self):
        """Data, exactly as provided by the user."""
        return self._data_raw

    @property
    def data(self) -> LayerDataProtocol:
        """Data, possibly in multiscale wrapper. Obeys LayerDataProtocol."""
        return self._data

    @data.setter
    def data(
        self, data: Union[LayerDataProtocol, Sequence[LayerDataProtocol]]
    ):
        self._data_raw = data
        # note, we don't support changing multiscale in an Image instance
        self._data = MultiScaleData(data) if self.multiscale else data  # type: ignore
        self._update_dims()
        self.events.data(value=self.data)
        if self._keep_auto_contrast:
            self.reset_contrast_limits()
        self._set_editable()

    def _get_ndim(self):
        """Determine number of dimensions of the layer."""
        return len(self.level_shapes[0])

    @property
    def _extent_data(self) -> np.ndarray:
        """Extent of layer in data coordinates.

        Returns
        -------
        extent_data : array, shape (2, D)
        """
        shape = self.level_shapes[0]
        return np.vstack([np.zeros(len(shape)), shape])

    @property
    def data_level(self):
        """int: Current level of multiscale, or 0 if image."""
        return self._data_level

    @data_level.setter
    def data_level(self, level):
        if self._data_level == level:
            return
        self._data_level = level
        self.refresh()

    @property
    def level_shapes(self):
        """array: Shapes of each level of the multiscale or just of image."""
        shapes = self.data.shapes if self.multiscale else [self.data.shape]
        if self.rgb:
            shapes = [s[:-1] for s in shapes]
        return np.array(shapes)

    @property
    def downsample_factors(self):
        """list: Downsample factors for each level of the multiscale."""
        return np.divide(self.level_shapes[0], self.level_shapes)

    @property
    def iso_threshold(self):
        """float: threshold for isosurface."""
        return self._iso_threshold

    @iso_threshold.setter
    def iso_threshold(self, value):
        self._iso_threshold = value
        self._update_thumbnail()
        self.events.iso_threshold()

    @property
    def attenuation(self):
        """float: attenuation rate for attenuated_mip rendering."""
        return self._attenuation

    @attenuation.setter
    def attenuation(self, value):
        self._attenuation = value
        self._update_thumbnail()
        self.events.attenuation()

    @property
    def interpolation(self):
        """Return current interpolation mode.

        Selects a preset interpolation mode in vispy that determines how volume
        is displayed.  Makes use of the two Texture2D interpolation methods and
        the available interpolation methods defined in
        vispy/gloo/glsl/misc/spatial_filters.frag

        Options include:
        'bessel', 'bicubic', 'bilinear', 'blackman', 'catrom', 'gaussian',
        'hamming', 'hanning', 'hermite', 'kaiser', 'lanczos', 'mitchell',
        'nearest', 'spline16', 'spline36'

        Returns
        -------
        str
            The current interpolation mode
        """
        return str(self._interpolation[self._ndisplay])

    @interpolation.setter
    def interpolation(self, interpolation):
        """Set current interpolation mode."""
        if self._ndisplay == 3:
            self._interpolation[self._ndisplay] = Interpolation3D(
                interpolation
            )
        else:
            self._interpolation[self._ndisplay] = Interpolation(interpolation)
        self.events.interpolation(value=self._interpolation[self._ndisplay])

    @property
    def depiction(self):
        """The current 3D depiction mode.

        Selects a preset depiction mode in vispy
            * volume: images are rendered as 3D volumes.
            * plane: images are rendered as 2D planes embedded in 3D.
                plane position, normal, and thickness are attributes of
                layer.plane which can be modified directly.
        """
        return str(self._depiction)

    @depiction.setter
    def depiction(self, depiction: Union[str, VolumeDepiction]):
        """Set the current 3D depiction mode."""
        self._depiction = VolumeDepiction(depiction)
        self._update_plane_callbacks()
        self.events.depiction()

    def _reset_plane_parameters(self):
        """Set plane attributes to something valid."""
        self.plane.position = np.array(self.data.shape) / 2
        self.plane.normal = (1, 0, 0)

    def _update_plane_callbacks(self):
        """Set plane callbacks depending on depiction mode."""
        plane_drag_callback_connected = (
            plane_drag_callback in self.mouse_drag_callbacks
        )
        double_click_callback_connected = (
            plane_double_click_callback in self.mouse_double_click_callbacks
        )
        if self.depiction == VolumeDepiction.VOLUME:
            if plane_drag_callback_connected:
                self.mouse_drag_callbacks.remove(plane_drag_callback)
            if double_click_callback_connected:
                self.mouse_double_click_callbacks.remove(
                    plane_double_click_callback
                )
        elif self.depiction == VolumeDepiction.PLANE:
            if not plane_drag_callback_connected:
                self.mouse_drag_callbacks.append(plane_drag_callback)
            if not double_click_callback_connected:
                self.mouse_double_click_callbacks.append(
                    plane_double_click_callback
                )

    @property
    def plane(self):
        return self._plane

    @plane.setter
    def plane(self, value: Union[dict, SlicingPlane]):
        self._plane.update(value)

    @property
    def mode(self) -> str:
        """str: Interactive mode

        Interactive mode. The normal, default mode is PAN_ZOOM, which
        allows for normal interactivity with the canvas.

        TRANSFORM allows for manipulation of the layer transform.
        """
        return str(self._mode)

    _drag_modes = {Mode.TRANSFORM: no_op, Mode.PAN_ZOOM: no_op}

    _move_modes = {
        Mode.TRANSFORM: no_op,
        Mode.PAN_ZOOM: no_op,
    }
    _cursor_modes = {
        Mode.TRANSFORM: 'standard',
        Mode.PAN_ZOOM: 'standard',
    }

    @mode.setter
    def mode(self, mode):
        mode, changed = self._mode_setter_helper(mode, Mode)
        if not changed:
            return
        assert mode is not None, mode

        if mode == Mode.PAN_ZOOM:
            self.help = ''
        else:
            self.help = trans._(
                'hold <space> to pan/zoom, hold <shift> to preserve aspect ratio and rotate in 45° increments'
            )

        self.events.mode(mode=mode)

    def _raw_to_displayed(self, raw):
        """Determine displayed image from raw image.

        For normal image layers, just return the actual image.

        Parameters
        ----------
        raw : array
            Raw array.

        Returns
        -------
        image : array
            Displayed array.
        """
        image = raw
        return image

    def _get_slice(self, request: LayerSliceRequest) -> LayerSliceResponse:
        LOGGER.debug('Image._get_slice : %s', request)
        slice_indices = self._get_slice_indices(request)

        data, tile_to_data = (
            self._get_slice_data_multi_scale(slice_indices, request)
            if self.multiscale
            else self._get_slice_data(slice_indices, request)
        )

        full_transform = self._transforms.simplified
        if tile_to_data is not None:
            full_transform = tile_to_data.compose(full_transform)

        dims_displayed = list(request.dims_displayed)
        transform = full_transform.set_slice(dims_displayed)
        if request.ndisplay == 2:
            transform = self._offset_2d_image_transform(
                transform, dims_displayed
            )

        # TODO: expand dims of data if ndisplay is 3 and ndim is 2.

        # TODO: downsample data if it exceeds GL texture max size.

        # TODO: fix thumbnail for partial view of multi-scale data
        thumbnail = self._make_thumbnail(data)

        return LayerSliceResponse(
            request=request,
            data=data,
            thumbnail=thumbnail,
            transform=transform,
        )

    def _offset_2d_image_transform(self, transform, dims_displayed) -> Affine:
        # Perform pixel offset to shift origin from top left corner
        # of pixel to center of pixel.
        # Note this offset is only required for array like data in
        # 2D.
        offset_matrix = self._data_to_world.set_slice(
            dims_displayed
        ).linear_matrix
        offset = -offset_matrix @ np.ones(offset_matrix.shape[1]) / 2
        # Convert NumPy axis ordering to VisPy axis ordering
        # and embed in full affine matrix
        affine_offset = np.eye(3)
        affine_offset[: len(offset), -1] = offset
        transform_matrix = transform.affine_matrix @ affine_offset
        return Affine(affine_matrix=transform_matrix)

    def _get_slice_data(
        self, slice_indices, request
    ) -> Tuple[np.ndarray, None]:
        if all(t == 1 for t in request.thickness_not_displayed):
            return np.asarray(self.data[slice_indices]), None

        image_slices = tuple(
            generate_thick_slices(
                slice_indices,
                request.thickness_not_displayed,
                self.data.shape,
                request.dims_not_displayed,
            )
        )

        return (
            project_slice(
                self.data,
                image_slices,
                axis=request.dims_not_displayed,
            ),
            None,
        )

    def _get_slice_data_multi_scale(
        self, slice_indices, request: LayerSliceRequest
    ) -> Tuple[np.ndarray, Affine]:
        if request.ndisplay == 3:
            warnings.warn(
                trans._(
                    'Multiscale rendering is only supported in 2D. In 3D, only the lowest resolution scale is displayed',
                    deferred=True,
                ),
                category=UserWarning,
            )
            # TODO: always infer data level and corners when slicing rather than mutating/relying on state
            # that is changed in Layer._update_draw
            self.data_level = len(self.data) - 1

        level = self.data_level
        indices = self._get_downsampled_indices(
            slice_indices, list(request.dims_not_displayed), level
        )

        scale = np.ones(self.ndim)
        for d in request.dims_displayed:
            scale[d] = self.downsample_factors[self.data_level][d]

        # This only needs to be a ScaleTranslate but different types
        # of transforms in a chain don't play nicely together right now.
        tile_to_data = Affine(scale=scale)

        if request.ndisplay == 2:
            for d in request.dims_displayed:
                indices[d] = slice(
                    self.corner_pixels[0, d],
                    self.corner_pixels[1, d],
                    1,
                )
            tile_to_data.translate = self.corner_pixels[0] * tile_to_data.scale
        return np.asarray(self.data[level][tuple(indices)]), tile_to_data

    def _get_downsampled_indices(self, indices, not_disp, level) -> np.ndarray:
        indices = np.array(indices)
        downsampled_indices = (
            indices[not_disp] / self.downsample_factors[level, not_disp]
        )
        downsampled_indices = np.round(
            downsampled_indices.astype(float)
        ).astype(int)
        downsampled_indices = np.clip(
            downsampled_indices,
            0,
            self.level_shapes[level, not_disp] - 1,
        )
        indices[not_disp] = downsampled_indices
        return indices

    def _are_slice_indices_out_of_range(self, indices, not_disp) -> bool:
        extent = self._extent_data
        return np.any(
            np.less(
                [indices[ax] for ax in not_disp],
                [extent[0, ax] for ax in not_disp],
            )
        ) or np.any(
            np.greater_equal(
                [indices[ax] for ax in not_disp],
                [extent[1, ax] for ax in not_disp],
            )
        )

    def _make_thumbnail(self, image):
        if self._ndisplay == 3 and self.ndim > 2:
            image = np.max(image, axis=0)

        # float16 not supported by ndi.zoom
        dtype = np.dtype(image.dtype)
        if dtype in [np.dtype(np.float16)]:
            image = image.astype(np.float32)

        raw_zoom_factor = np.divide(
            self._thumbnail_shape[:2], image.shape[:2]
        ).min()
        new_shape = np.clip(
            raw_zoom_factor * np.array(image.shape[:2]),
            1,  # smallest side should be 1 pixel wide
            self._thumbnail_shape[:2],
        )
        zoom_factor = tuple(new_shape / image.shape[:2])
        if self.rgb:
            # warning filter can be removed with scipy 1.4
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                downsampled = ndi.zoom(
                    image, zoom_factor + (1,), prefilter=False, order=0
                )
            if image.shape[2] == 4:  # image is RGBA
                colormapped = np.copy(downsampled)
                colormapped[..., 3] = downsampled[..., 3] * self.opacity
                if downsampled.dtype == np.uint8:
                    colormapped = colormapped.astype(np.uint8)
            else:  # image is RGB
                if downsampled.dtype == np.uint8:
                    alpha = np.full(
                        downsampled.shape[:2] + (1,),
                        int(255 * self.opacity),
                        dtype=np.uint8,
                    )
                else:
                    alpha = np.full(downsampled.shape[:2] + (1,), self.opacity)
                colormapped = np.concatenate([downsampled, alpha], axis=2)
        else:
            # warning filter can be removed with scipy 1.4
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                downsampled = ndi.zoom(
                    image, zoom_factor, prefilter=False, order=0
                )
            low, high = self.contrast_limits
            downsampled = np.clip(downsampled, low, high)
            color_range = high - low
            if color_range != 0:
                downsampled = (downsampled - low) / color_range
            downsampled = downsampled**self.gamma
            color_array = self.colormap.map(downsampled.ravel())
            colormapped = color_array.reshape(downsampled.shape + (4,))
            colormapped[..., 3] *= self.opacity
        return colormapped

    def _update_thumbnail(self):
        """Update thumbnail with current image data and colormap."""
        # TODO: return thumbnail when slicing instead of updating in-place.
        pass

    def _get_value(self, position):
        """Value of the data at a position in data coordinates.

        Parameters
        ----------
        position : tuple
            Position in data coordinates.

        Returns
        -------
        value : tuple
            Value of the data.
        """
        # TODO: this depends on the sliced state, so do nothing for now.
        return None

    def _get_offset_data_position(self, position: List[float]) -> List[float]:
        """Adjust position for offset between viewer and data coordinates.

        VisPy considers the coordinate system origin to be the canvas corner,
        while napari considers the origin to be the **center** of the corner
        pixel. To get the correct value under the mouse cursor, we need to
        shift the position by 0.5 pixels on each axis.
        """
        return [p + 0.5 for p in position]


class Image(_ImageBase):
    @property
    def rendering(self):
        """Return current rendering mode.

        Selects a preset rendering mode in vispy that determines how
        volume is displayed.  Options include:

        * ``translucent``: voxel colors are blended along the view ray until
            the result is opaque.
        * ``mip``: maximum intensity projection. Cast a ray and display the
            maximum value that was encountered.
        * ``minip``: minimum intensity projection. Cast a ray and display the
            minimum value that was encountered.
        * ``attenuated_mip``: attenuated maximum intensity projection. Cast a
            ray and attenuate values based on integral of encountered values,
            display the maximum value that was encountered after attenuation.
            This will make nearer objects appear more prominent.
        * ``additive``: voxel colors are added along the view ray until
            the result is saturated.
        * ``iso``: isosurface. Cast a ray until a certain threshold is
            encountered. At that location, lighning calculations are
            performed to give the visual appearance of a surface.
        * ``average``: average intensity projection. Cast a ray and display the
            average of values that were encountered.

        Returns
        -------
        str
            The current rendering mode
        """
        return str(self._rendering)

    @rendering.setter
    def rendering(self, rendering):
        self._rendering = ImageRendering(rendering)
        self.events.rendering()

    def _get_state(self):
        """Get dictionary of layer state.

        Returns
        -------
        state : dict
            Dictionary of layer state.
        """
        state = self._get_base_state()
        state.update(
            {
                'rgb': self.rgb,
                'multiscale': self.multiscale,
                'colormap': self.colormap.name,
                'contrast_limits': self.contrast_limits,
                'interpolation': self.interpolation,
                'rendering': self.rendering,
                'depiction': self.depiction,
                'plane': self.plane.dict(),
                'iso_threshold': self.iso_threshold,
                'attenuation': self.attenuation,
                'gamma': self.gamma,
                'data': self.data,
            }
        )
        return state


if config.async_octree:
    from ..image.experimental.octree_image import _OctreeImageBase

    class Image(Image, _OctreeImageBase):
        pass


class _weakref_hide:
    def __init__(self, obj):
        import weakref

        self.obj = weakref.ref(obj)

    def _raw_to_displayed(self, *args, **kwarg):
        return self.obj()._raw_to_displayed(*args, **kwarg)


def generate_thick_slices(
    slice_indices, slice_thicknesses, data_shape, dims_not_displayed
):
    for i in range(len(data_shape)):
        if i in dims_not_displayed:
            half_thick = max(ceil(slice_thicknesses[i]), 1) / 2
            idx = slice_indices[i]
            # round up always with ceil, this way the extremes become:
            # - slice(i, i+1) for min thickness
            # - slice(0, shape+1) for max thickness
            sl_start = max(0, ceil(idx - half_thick))
            sl_end = min(ceil(idx + half_thick), data_shape[i])
            yield slice(sl_start, sl_end)
        else:
            yield slice_indices[i]


def project_slice(data, slices, axis):
    return np.mean(data[slices], tuple(axis))
