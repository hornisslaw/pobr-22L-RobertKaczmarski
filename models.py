class SegmentModel:
    def __init__(self, wh_ratio, inv_moment_mins, inv_moment_maxs) -> None:
        self.wh_ratio = wh_ratio
        self.inv_moment_mins = inv_moment_mins
        self.inv_moment_maxs = inv_moment_maxs

    def check_invariant_moment(self, index, invariant_moment):
        return (
            self.inv_moment_mins[index]
            <= invariant_moment
            <= self.inv_moment_maxs[index]
        )

    def get_number_or_segments(self):
        return len(self.inv_moment_mins)
