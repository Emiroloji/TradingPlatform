import main


def test_scheduler_has_daily_and_monthly_jobs_per_market(cfg):
    scheduler = main.build_scheduler(cfg)
    ids = {j.id for j in scheduler.get_jobs()}
    for market in cfg.enabled_markets():
        assert {f"daily_{market}", f"retrain_{market}"} <= ids
    daily = scheduler.get_job("daily_bist").trigger
    fields = {f.name: str(f) for f in daily.fields}
    assert fields["day_of_week"] == "mon-fri" and fields["hour"] == "18" and fields["minute"] == "45"
    assert str(daily.timezone) == cfg.schedule.timezone
    assert scheduler._executors["default"]._pool._max_workers == 1  # jobs never overlap


def test_isolated_job_crash_sends_alert(cfg, monkeypatch):
    import sys

    import report.telegram_bot

    alerts = []
    monkeypatch.setattr(report.telegram_bot, "send_alert", lambda text, cfg: alerts.append(text))
    main._run_isolated(str.upper, "bist")  # a child that returns normally: no alert
    assert alerts == []
    main._run_isolated(sys.exit, "bist")  # a child that dies with a non-zero code, as a killed one would
    assert len(alerts) == 1 and alerts[0].startswith("BIST") and "çıkış kodu 1" in alerts[0]
