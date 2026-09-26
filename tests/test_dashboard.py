from pathlib import Path

import pytest

from config import load_config

APP = str(Path(__file__).resolve().parents[1] / "report" / "dashboard.py")


@pytest.mark.skipif(not load_config().storage_path.joinpath("features", "bist.parquet").exists(), reason="needs a features run")
def test_dashboard_renders_all_tabs_without_errors():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 4


def test_password_gate_blocks_until_correct(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("DASHBOARD_PASSWORD", "dogru-sifre")
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception and len(at.tabs) == 0  # nothing but the password box
    at.text_input[0].input("yanlis").run()
    assert len(at.tabs) == 0 and at.error
    at.text_input[0].input("dogru-sifre").run()
    assert len(at.tabs) == 4
