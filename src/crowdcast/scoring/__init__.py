"""Turn regional holiday calendars into park-level scores.

    national_score(park, day) = sum_r weight(park, r, national) * national_holiday(r, day)
    school_score(park, day)   = sum_r weight(park, r, school)   * school_holiday(r, day)

Weights sum to 1 per park and type, so both scores lie in [0, 1]. See ``weights`` for how the weights are estimated.
"""

from crowdcast.scoring.daily import CalendarMatrices, DailyScores, compute_daily_scores

__all__ = ["CalendarMatrices", "DailyScores", "compute_daily_scores"]
