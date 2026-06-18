from enum import IntEnum

# The internal solver code works with these plain integer constants directly because they
# are convenient as scalar arguments to tensor operations such as `torch.where`. The
# `Status` enum below mirrors them as a more user-friendly, public alternative that can,
# for example, give you the name of an error code.
SUCCESS = 0
GENERAL_ERROR = 1
REACHED_DT_MIN = 2
REACHED_MAX_STEPS = 3
INFINITE_NORM = 4


class Status(IntEnum):
    """A joint collection of all possible solver and step size controller status codes.

    Any status other than `Status.SUCCESS` signifies some type of abnormal condition.
    """

    SUCCESS = SUCCESS
    GENERAL_ERROR = GENERAL_ERROR
    REACHED_DT_MIN = REACHED_DT_MIN
    REACHED_MAX_STEPS = REACHED_MAX_STEPS

    # The norm of the error ratio turned out infinite which points towards some bad
    # problems such as infinite or NaN y and the solving should be cancelled.
    INFINITE_NORM = INFINITE_NORM
