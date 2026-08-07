"""Exception types for the pipeline."""


class DataContractError(RuntimeError):
    """Raised when input data violates the documented format contract.

    Always carries an actionable message: what was expected, what was found,
    and which file caused it.
    """
