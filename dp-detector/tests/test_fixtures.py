from pathlib import Path
from bs4 import BeautifulSoup

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
EXPECTED = [
    "precheck_consent", "hidden_cancel", "forced_continuity",
    "confirm_shaming", "disguised_button", "deep_cancel_flow", "easy_cancel_flow",
]

def test_all_fixtures_exist_and_are_valid_html():
    for name in EXPECTED:
        path = FIXTURES_DIR / f"{name}.html"
        assert path.exists(), f"missing fixture {path}"
        soup = BeautifulSoup(path.read_text(), "html.parser")
        assert soup.body is not None
        assert soup.body.get("data-fixture-id") == name
