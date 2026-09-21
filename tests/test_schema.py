import pandas as pd
import pytest

from crowdcast.data.schema import CROWD_CALENDAR, SchemaError, validate
from crowdcast.data.synthetic import make_synthetic


def test_valid_table_passes_unchanged():
    crowd = make_synthetic(2, end="2019-03-01").crowd
    assert validate(crowd, CROWD_CALENDAR) is crowd


def test_reports_every_problem_at_once():
    bad = pd.DataFrame(
        {"park_id": [1, 1], "date": ["2020-01-01"] * 2, "status": ["open", "weird"], "crowd_percent": [5, 500]}
    )
    with pytest.raises(SchemaError) as err:
        validate(bad, CROWD_CALENDAR)
    msg = str(err.value)
    assert "missing columns" in msg and "duplicate rows" in msg and "unexpected values" in msg and "outside" in msg
