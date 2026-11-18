class ScanCancelledError(RuntimeError):
    """Raised when a cooperative dataset scan is cancelled."""


class ScanAlreadyRunningError(RuntimeError):
    """Raised when the same dataset is submitted more than once."""


class DatasetConfigurationError(ValueError):
    """Raised when a dataset cannot be read with the supplied configuration."""

