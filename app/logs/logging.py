from typing import TypeVar
import loguru
from app.configs import config

log_level = "DEBUG"
log_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | "
    "<level>{level: <8}</level> | "
    "<yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"
)

_T_loguru_logger = TypeVar("_T_loguru_logger", bound=loguru._logger.Logger)

def setup_logger(
    use_log_file: bool = True,
    file: str = config.logging_file,
    rotation: str = "50 MB",
    retention: str = "30 days",
) -> _T_loguru_logger:
    loguru.logger.remove()
    if use_log_file:
        loguru.logger.add(
            file,
            level=log_level,
            format=log_format,
            colorize=False,
            rotation=rotation,
            retention=retention,
            backtrace=True,
            diagnose=True,
        )

    loguru.logger.add(
        sink=lambda msg: print(msg, end=""),
        level=log_level,
        format=log_format,
        colorize=True
    )

    return loguru.logger

log = setup_logger()