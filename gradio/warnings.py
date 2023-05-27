from functools import wraps
import logging
import warnings

logger = logging.getLogger(__name__)


def warn_kwargs(cls, kwargs):
    for k in kwargs:
        logger.warning(
            f"Warning: {cls.__name__} does not accept the {k} parameter. This parameter will be ignored."
        )


def warn_unimplemented(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"Function {fn.__name__} is not implemented for {args[0].__class__.__name__} components."
        )
        return fn(*args, **kwargs)

    return wrapper
