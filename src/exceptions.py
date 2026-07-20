class ObstacleCollisionError(Exception):
    """Exception for attempting to drive into an obstacle"""

    def __init__(self, *args) -> None:  # noqa: D107
        super().__init__(*args)


class TargetNotReachedError(Exception):
    """Exception for cases where the target point was not reached"""

    def __init__(self, *args) -> None:  # noqa: D107
        super().__init__(*args)
