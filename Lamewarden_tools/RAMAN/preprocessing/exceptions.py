class SGFilterError(Exception):
    """Exception raised for errors in the Savitzky-Golay filtering operation."""
    pass

class DespikingError(Exception):
    """Exception raised for errors in the despiking process."""
    pass

class BaselineCorrectionError(Exception):
    """Exception raised for errors during baseline correction."""
    pass

class CalibrationError(Exception):
    """Exception raised for errors during baseline correction."""
    pass