"""Napari Configuration.
"""
import os
import warnings

from napari.utils.translations import trans


def _set(env_var: str) -> bool:
    """Return True if the env variable is set and non-zero.

    Returns
    -------
    bool
        True if the env var was set to a non-zero value.
    """
    return os.getenv(env_var) not in [None, "0"]


"""
Experimental Features

Async Loading
-------------
Deprecated.

Octree Rendering
----------------
Deprecated.

Shared Memory Server
--------------------
Experimental shared memory service. Only enabled if NAPARI_MON is set to
the path of a config file. See this PR for more info:
https://github.com/napari/napari/pull/1909.
"""


def _warn_about_deprecated_attribute(name) -> None:
    warnings.warn(
        trans._(
            '{name} is deprecated from napari version 0.5 and will be removed in the later version.',
            name=name,
        ),
        DeprecationWarning,
        stacklevel=2,
    )


# Handle old async/octree deprecated attributes by returning their
# fixed values in the module level __getattr__
# https://peps.python.org/pep-0562/
# Other module attributes are defined as normal.
def __getattr__(name):
    if name == 'octree_config':
        _warn_about_deprecated_attribute(name)
        return None
    elif name in ('async_loading', 'async_octree'):
        _warn_about_deprecated_attribute(name)
        # For async_loading, we could get the value of the remaining
        # async setting. We do not because that is dynamic, so will not
        # handle an import of the form
        #
        # `from napari.utils.config import async_loading`
        #
        # consistently. Instead, we let this attribute effectively
        # refer to the old async which is always off in napari now.
        return False


# Shared Memory Server
monitor = _set("NAPARI_MON")

"""
Other Config Options
"""
# Added this temporarily for octree debugging. The welcome visual causes
# breakpoints to hit in image visual code. It's easier if we don't show it.
allow_welcome_visual = True
