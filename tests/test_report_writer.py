from ai.gemini_client import GeminiError
from ai.report_writer import SignalRow, check_commentary, facts, template_commentary, write_commentary

SIGNAL = SignalRow(
    ticker="THYAO", name="Türk Hava Yolları", total_score=82, accumulation_score=76, prob=None, regime="bull",
    news_status="haber kontrolü yapıldı", news_summary="Olumlu sipariş haberleri.", entry_ref=312.5, stop=298.25,
    position_size=70, bt_trades=118, bt_win_rate=0.64, bt_expectancy=0.033, bt_max_dd=-0.12,
)


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply

    def generate_text(self, prompt, system):
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply

    def generate_json(self, prompt, system, schema):
        raise NotImplementedError


def test_facts_format():
    f = facts(SIGNAL)
    assert "Toplam skor: 82/100" in f
    assert any("expectancy +%3,3" in x and "%64,0" in x and "-%12,0" in x for x in f)
    assert any("312,50" in x and "298,25" in x for x in f)
    assert any("kullanılmıyor" in x for x in f)


def test_accepts_text_that_reuses_given_numbers(cfg):
    text = "THYAO'da toplam skor 82/100, toplama skoru 76/100. Geçmişte 118 işlemde %64,0 kazanma oranı görülmüş."
    assert check_commentary(text, facts(SIGNAL), cfg) == []
    assert write_commentary(SIGNAL, FakeLLM(text), cfg) == text


def test_invented_number_falls_back_to_template(cfg):
    text = "Hedef fiyat 350 olabilir."
    assert check_commentary(text, facts(SIGNAL), cfg)
    assert write_commentary(SIGNAL, FakeLLM(text), cfg) == template_commentary(SIGNAL)


def test_certainty_wording_falls_back(cfg):
    text = "Toplam skor 82/100 ile kesinlikle yükselir."
    assert any("banned" in p for p in check_commentary(text, facts(SIGNAL), cfg))
    assert write_commentary(SIGNAL, FakeLLM(text), cfg) == template_commentary(SIGNAL)


def test_api_error_falls_back(cfg):
    assert write_commentary(SIGNAL, FakeLLM(GeminiError("x")), cfg) == template_commentary(SIGNAL)


def test_template_uses_only_given_numbers(cfg):
    assert check_commentary(template_commentary(SIGNAL), facts(SIGNAL), cfg) == []


def test_small_expectancy_keeps_two_decimals_and_no_double_period():
    s = SIGNAL.model_copy(update={"bt_expectancy": -0.0003, "news_summary": "Olumlu."})
    f = facts(s)
    assert any("-%0,03" in x for x in f)
    assert not any(".." in x for x in f)
