import json
from datetime import datetime, timezone

import pandas as pd

from ai.gemini_client import GeminiError
from ai.news import NewsItem
from ai.sentiment import analyze_news, log_veto

ITEMS = [NewsItem("1", "yfinance", "Şirkete SPK soruşturması", "", "", datetime(2026, 9, 25, tzinfo=timezone.utc))]


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def generate_json(self, prompt, system, schema):
        self.prompts.append((prompt, system))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply

    def generate_text(self, prompt, system):
        raise NotImplementedError


def _json(**kw):
    base = {"duyarlilik": 0.2, "veto": False, "veto_nedeni": "", "ozet": "Olumlu gelişmeler."}
    return json.dumps(base | kw, ensure_ascii=False)


def test_ok_passes(cfg):
    r = analyze_news("AAA", ITEMS, FakeLLM(_json()), cfg)
    assert r.status == "ok" and r.passes and r.sentiment == 0.2


def test_veto_blocks_and_keeps_reason(cfg):
    r = analyze_news("AAA", ITEMS, FakeLLM(_json(veto=True, veto_nedeni="SPK soruşturması", duyarlilik=-0.8)), cfg)
    assert r.veto and not r.passes and r.veto_reason == "SPK soruşturması"


def test_no_news_is_not_an_error(cfg):
    llm = FakeLLM(_json())
    r = analyze_news("AAA", [], llm, cfg)
    assert r.status == "no_news" and r.passes
    assert llm.prompts == []  # no API call without news


def test_api_error_blocks_signal_but_does_not_raise(cfg):
    r = analyze_news("AAA", ITEMS, FakeLLM(GeminiError("down")), cfg)
    assert r.status == "check_failed" and not r.passes
    assert r.status_text == "haber kontrolü yapılamadı"


def test_invalid_json_or_schema_is_no_data(cfg):
    for bad in ["not json", _json(duyarlilik=3), json.dumps({"veto": False}), _json(veto=True, veto_nedeni=" ")]:
        r = analyze_news("AAA", ITEMS, FakeLLM(bad), cfg)
        assert r.status == "invalid_response" and not r.passes, bad


def test_prompt_contains_rules_and_veto_categories(cfg):
    llm = FakeLLM(_json())
    analyze_news("AAA", ITEMS, llm, cfg, name="Örnek A.Ş.")
    prompt, system = llm.prompts[0]
    assert "Şirkete SPK soruşturması" in prompt and "Örnek A.Ş." in prompt
    assert "tahmini YAPMA" in system
    assert all(c in system for c in cfg.gemini.veto_categories)


def test_veto_log(cfg, tmp_path):
    local = cfg.model_copy(update={"data": cfg.data.model_copy(update={"storage_dir": str(tmp_path)})})
    r = analyze_news("AAA", ITEMS, FakeLLM(_json(veto=True, veto_nedeni="SPK soruşturması")), local)
    log_veto(r, ITEMS, local, datetime(2026, 9, 26))
    log_veto(r, ITEMS, local, datetime(2026, 9, 27))
    log = pd.read_csv(tmp_path / "logs" / "veto_log.csv")
    assert len(log) == 2 and log["reason"].iloc[0] == "SPK soruşturması"


def test_check_candidates_all_sources_failed_blocks(cfg, monkeypatch, tmp_path):
    import ai.news as news_mod
    from ai.sentiment import check_candidates

    local = cfg.model_copy(update={"data": cfg.data.model_copy(update={"storage_dir": str(tmp_path)})})
    monkeypatch.setattr(news_mod, "collect_news", lambda *a: ([], ["yfinance", "google_news_rss"]))
    out = check_candidates(pd.DataFrame({"ticker": ["AAA.IS"]}), {}, FakeLLM(_json()), local, datetime(2026, 9, 26), "hl=tr")
    assert out.loc[0, "news_status"] == "haber kontrolü yapılamadı" and not out.loc[0, "gemini_passes"]


def test_check_candidates_veto_logged(cfg, monkeypatch, tmp_path):
    import ai.news as news_mod
    from ai.sentiment import check_candidates

    local = cfg.model_copy(update={"data": cfg.data.model_copy(update={"storage_dir": str(tmp_path)})})
    monkeypatch.setattr(news_mod, "collect_news", lambda *a: (ITEMS, []))
    llm = FakeLLM(_json(veto=True, veto_nedeni="SPK soruşturması", duyarlilik=-0.9))
    out = check_candidates(pd.DataFrame({"ticker": ["AAA.IS"]}), {"AAA.IS": "Örnek"}, llm, local, datetime(2026, 9, 26), "hl=tr")
    assert out.loc[0, "gemini_veto"] and out.loc[0, "veto_reason"] == "SPK soruşturması"
    assert (tmp_path / "logs" / "veto_log.csv").exists()


def test_summary_numbers_must_come_from_the_news(cfg):
    items = [NewsItem("1", "yfinance", "Şirket 750 milyon TL geri alım yapacak", "", "", datetime(2026, 9, 25, tzinfo=timezone.utc))]
    quoted = analyze_news("AAA", items, FakeLLM(_json(ozet="750 milyon TL geri alım.")), cfg)
    assert quoted.summary == "750 milyon TL geri alım."
    invented = analyze_news("AAA", items, FakeLLM(_json(ozet="Hisse 900 TL olabilir.")), cfg)
    assert invented.summary == "" and invented.status == "ok"  # decision kept, invented text dropped
