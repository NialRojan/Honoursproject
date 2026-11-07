import numpy as np
import pandas as pd

from eval.splits import normalise_template, make_groups, grouped_folds, random_folds


def test_normalise_template_collapses_numbers_and_punctuation():
    assert normalise_template("Only 3 left!") == normalise_template("only 17 LEFT")
    assert normalise_template("Hurry, only 3 left in stock.") == "hurry only # left in stock"


def test_groups_link_rows_sharing_page_or_template():
    df = pd.DataFrame({
        "page_id": [1, 2, 3, 4],
        "text": ["Only 3 left", "Only 9 left", "Totally different", "Another one"],
    })
    g = make_groups(df)
    assert g[0] == g[1]          # same template
    assert len({g[2], g[3]}) == 2 and g[2] != g[0]
    df2 = pd.DataFrame({"page_id": [7, 7, 8], "text": ["a b", "c d", "e f"]})
    g2 = make_groups(df2)
    assert g2[0] == g2[1] and g2[0] != g2[2]   # same page


def test_grouped_folds_never_split_a_group_and_cover_every_row_once():
    rng = np.random.default_rng(0)
    groups = rng.integers(0, 40, size=300)
    y = rng.integers(0, 2, size=300)
    folds = grouped_folds(y, groups, k=5, seed=1)
    seen = []
    for tr, te in folds:
        assert set(groups[tr]).isdisjoint(set(groups[te]))
        seen.extend(te.tolist())
    assert sorted(seen) == list(range(300))


def test_random_folds_cover_every_row_once_and_are_seeded():
    y = np.array([0, 1] * 50)
    a = random_folds(y, k=5, seed=3)
    b = random_folds(y, k=5, seed=3)
    assert [te.tolist() for _, te in a] == [te.tolist() for _, te in b]
    assert sorted(i for _, te in a for i in te.tolist()) == list(range(100))
