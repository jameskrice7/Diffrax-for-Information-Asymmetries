"""Package logging.

Follows the library convention of attaching a ``NullHandler`` to the root
``finax`` logger, so importing finax never configures logging for the
application that imports it. Call :func:`set_level` to opt in to output.
"""

from __future__ import annotations

import logging

__all__ = ["get_logger", "set_level"]

_ROOT = "finax"
logging.getLogger(_ROOT).addHandler(logging.NullHandler())


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger under the ``finax`` namespace.

    Examples
    --------
    >>> get_logger("models").name
    'finax.models'
    >>> get_logger().name
    'finax'
    """
    if name is None or name == _ROOT:
        return logging.getLogger(_ROOT)
    return logging.getLogger(f"{_ROOT}.{name.removeprefix(_ROOT + '.')}")


def set_level(level: int | str = logging.INFO, *, stream: bool = True) -> None:
    """Enable finax logging at ``level``.

    Examples
    --------
    >>> import logging
    >>> set_level(logging.WARNING)
    >>> get_logger().level == logging.WARNING
    True
    """
    logger = logging.getLogger(_ROOT)
    logger.setLevel(level)
    if stream and not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        )
        logger.addHandler(handler)
