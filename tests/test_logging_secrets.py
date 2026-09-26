import logging

import main


def test_secrets_are_masked_in_any_logger(monkeypatch, caplog):
    secret = "123456:ABCDEFsecretTOKEN"
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", secret)
    handler_filter = main.RedactSecrets()
    record = logging.LogRecord("httpx", logging.INFO, "", 0, "POST https://api.telegram.org/bot%s/sendMessage", (secret,), None)
    handler_filter.filter(record)
    assert secret not in record.getMessage() and "***" in record.getMessage()


def test_http_loggers_quiet_after_setup(cfg, tmp_path):
    local = cfg.model_copy(update={"data": cfg.data.model_copy(update={"storage_dir": str(tmp_path)})})
    main.setup_logging(local)
    assert logging.getLogger("httpx").getEffectiveLevel() >= logging.WARNING
