import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from sim import bracket_order, playoffs, regular_season, round_robin, validate


@pytest.mark.parametrize("n", [4, 5, 14])
def test_round_robin_covers_every_pair(n):
    games = [tuple(sorted(p)) for rnd in round_robin(n) for p in rnd if min(p) >= 0]
    assert len(games) == len(set(games)) == len(list(combinations(range(n), 2)))


def test_bracket_order():
    assert bracket_order(8) == [1, 8, 4, 5, 2, 7, 3, 6]


def test_regular_season_strongest_team_first():
    weekly = np.tile(np.arange(4.0)[:, None], (1, 3))
    rank, wins = regular_season(weekly, 3)
    assert rank.tolist() == [3, 2, 1, 0] and wins[3] == 3


def test_regular_season_ties():
    rank, wins = regular_season(np.ones((4, 3)), 3)
    assert wins.tolist() == [1.5, 1.5, 1.5, 1.5]


def test_playoffs_six_teams_with_byes():
    weekly = np.tile(np.arange(6.0)[:, None], (1, 3))
    assert playoffs([0, 1, 2, 3, 4, 5], weekly, [0, 1, 2]) == 5
    assert playoffs([5, 4, 3, 2, 1, 0], weekly, [0, 1, 2]) == 5
    with pytest.raises(ValueError):
        playoffs(list(range(8)), np.ones((8, 2)), [0, 1])


def test_validate():
    ok = {"teams": 14, "playoff_teams": 6, "playoff_weeks": 3}
    validate(ok, 14)
    for bad, slot in [(ok, 15), (ok, 0), ({**ok, "playoff_teams": 9}, 1), ({**ok, "playoff_teams": 15}, 1)]:
        with pytest.raises(ValueError):
            validate(bad, slot)
