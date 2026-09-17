"""Exceptions raised by backend services."""


class PatientNotFoundError(LookupError):
    """Raised when a requested patient does not exist."""


class TreatmentPlanNotFoundError(LookupError):
    """Raised when a requested treatment plan does not exist."""


class InvalidPatientDataError(ValueError):
    """Raised when patient or treatment plan data is invalid."""


class MissingModelFeaturesError(ValueError):
    """Raised when persisted data lacks required model features."""
