from datetime import date

from ai.report_writer import SignalRow
from report.telegram_bot import DISCLAIMER, DailyContext, build_daily_message, send, split_message

ROW = SignalRow(
    ticker="THYAO", name="THY", total_score=82, accumulation_score=76, prob=None, regime="bull",
    news_status="haber kontrolü yapıldı", news_summary="Olumlu.", entry_ref=312.5, stop=298.25,
    position_size=70, bt_trades=118, bt_win_rate=0.64, bt_expectancy=0.033, bt_max_dd=-0.12,
)


def _ctx(**kw):
    base = dict(as_of=date(2026, 9, 25), market="bist", regime="bull", funnel=[("evren", 88), ("likidite", 80)],
                setup={"period": "2023-07-01 → 2026-09-25"}, setup_note="ok", ml_note="onaylı model yok")
    return DailyContext(**(base | kw))


def test_signal_message_has_every_required_field():
    text = build_daily_message(_ctx(signals=[(ROW, "Yorum metni.")]))
    for needle in ["THYAO — Temkinli Sinyal", "Toplam skor: 82/100", "Toplama skoru: 76/100", "ML olasılığı",
                   "Piyasa rejimi: yükseliş", "Gemini haber kontrolü", "118 işlem", "Expectancy: +%3,30",
                   "Max drawdown: -%12,0", "Önerilen stop: 298,25", "Pozisyon: 70 adet", "Yorum metni."]:
        assert needle in text, needle
    assert text.rstrip().endswith(DISCLAIMER)


def test_no_signal_day_explains_why():
    text = build_daily_message(_ctx(regime="bear", setup_note="kurulumun OOS expectancy'si pozitif değil",
                                    vetoed=[("RYGYO.IS", "devre kesici")]))
    assert "Bugün temkinli sinyal yok." in text
    assert "Piyasa rejimi (USD bazlı endeks): düşüş" in text
    assert "pozitif değil" in text and "RYGYO.IS: devre kesici" in text
    assert "- likidite: 80" in text


def test_split_respects_limit():
    text = "\n".join(f"satır {i} " + "x" * 50 for i in range(200))
    chunks = split_message(text, 500)
    assert all(len(c) <= 500 for c in chunks)
    assert "\n".join(chunks).replace("\n", "") == text.replace("\n", "")


def test_send_without_token_is_skipped(cfg, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "")
    monkeypatch.setattr("report.telegram_bot.load_dotenv", lambda *a, **k: None)
    assert send("merhaba", cfg) is False


def test_observation_section_is_labelled_not_live():
    text = build_daily_message(_ctx(observations={"kurumsal_toplama": ["AAPL", "MSFT"]},
                                    observation_perf={"kurumsal_toplama": {"signals": 2, "closed": 0}}))
    assert "Gözlem modu — kurumsal_toplama (CANLI SİNYAL DEĞİL" in text
    assert "AAPL, MSFT" in text and "Bugün temkinli sinyal yok." in text
