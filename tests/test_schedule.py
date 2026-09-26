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
