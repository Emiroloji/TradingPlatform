"""Backtest metrics (KURALLAR §4, §8). All returns are net of costs; relative = vs benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import Config


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough fall of the equity curve (negative number, 0 if none)."""
    if equity.empty:
        return float("nan")
    return float((equity / equity.cummax() - 1).min())


def compute_metrics(
    trades: pd.DataFrame, benchmark: pd.DataFrame, equity: pd.Series | None = None, cfg: Config | None = None
) -> dict:
    """Per-trade statistics (needs `return_net`, `excess_return` columns) plus, with `equity`,
    portfolio drawdown and total return against the benchmark over the same dates."""
    n = len(trades)
    out: dict = {"trades": n}
    if cfg is not None:
        out["insufficient_data"] = n < cfg.signal.min_backtest_trades
    if n == 0:
        return out

    r = trades["return_net"]
    wins, losses = r[r > 0], r[r <= 0]
    win_rate = len(wins) / n
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = losses.mean() if len(losses) else 0.0
    gross_profit = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    gross_loss = -trades.loc[trades["pnl"] < 0, "pnl"].sum()
    out |= {
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": win_rate * avg_win + (1 - win_rate) * avg_loss,  # = mean net return per trade
        "expectancy_excess": trades["excess_return"].mean(),
        "beat_benchmark_rate": (trades["excess_return"] > 0).mean(),
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
        "holding_days_p25": trades["holding_days"].quantile(0.25),
        "holding_days_median": trades["holding_days"].median(),
        "holding_days_p75": trades["holding_days"].quantile(0.75),
        "exit_reasons": trades["exit_reason"].value_counts().to_dict(),
    }

    if equity is not None and not equity.empty:
        b = benchmark.set_index("date")["close"].sort_index()
        bench_start, bench_end = b.asof(equity.index[0]), b.asof(equity.index[-1])
        out |= {
            "max_drawdown": max_drawdown(equity),
            "total_return": equity.iloc[-1] / equity.iloc[0] - 1,
            "benchmark_return": bench_end / bench_start - 1,
        }
        out["relative_return"] = out["total_return"] - out["benchmark_return"]
        bench_curve = b.reindex(equity.index, method="ffill")
        out["benchmark_max_drawdown"] = max_drawdown(bench_curve)
    return out


def breakdown(trades: pd.DataFrame, benchmark: pd.DataFrame, by: pd.Series, cfg: Config) -> pd.DataFrame:
    """compute_metrics per group (e.g. signal year or regime), one row per group."""
    rows = {}
    for key, group in trades.groupby(by.values):
        m = compute_metrics(group, benchmark, cfg=cfg)
        m.pop("exit_reasons", None)
        rows[key] = m
    return pd.DataFrame.from_dict(rows, orient="index").replace([np.inf], np.nan)


# ---------------------------------------------------------------- report (Turkish, KURALLAR §8)

_LABELS = [
    ("trades", "İşlem sayısı", "{:.0f}"),
    ("win_rate", "Kazanma oranı", "{:.1%}"),
    ("avg_win", "Ort. kazanç", "{:+.2%}"),
    ("avg_loss", "Ort. kayıp", "{:+.2%}"),
    ("expectancy", "Expectancy / işlem (net)", "{:+.2%}"),
    ("expectancy_excess", "Expectancy / işlem (endekse göre)", "{:+.2%}"),
    ("beat_benchmark_rate", "Endeksi yenme oranı", "{:.1%}"),
    ("profit_factor", "Profit factor", "{:.2f}"),
    ("max_drawdown", "Max drawdown (portföy)", "{:.1%}"),
    ("total_return", "Toplam getiri (portföy)", "{:+.1%}"),
    ("benchmark_return", "Endeks getirisi (aynı dönem)", "{:+.1%}"),
    ("relative_return", "Relatif getiri", "{:+.1%}"),
    ("benchmark_max_drawdown", "Endeks max drawdown", "{:.1%}"),
    ("holding_days_median", "Medyan tutma süresi (gün)", "{:.0f}"),
]


def _fmt(value, pattern: str) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return pattern.format(value)


def metrics_table(columns: dict[str, dict]) -> str:
    """Markdown table: one row per metric, one column per run."""
    lines = ["| Metrik | " + " | ".join(columns) + " |", "|---|" + "---|" * len(columns)]
    for key, label, pattern in _LABELS:
        lines.append(f"| {label} | " + " | ".join(_fmt(m.get(key), pattern) for m in columns.values()) + " |")
    flags = ["**yetersiz veri**" if m.get("insufficient_data") else "yeterli" for m in columns.values()]
    lines.append("| Örneklem (≥30 işlem) | " + " | ".join(flags) + " |")
    return "\n".join(lines)


def breakdown_table(table: pd.DataFrame, title: str) -> str:
    if table.empty:
        return f"### {title}\n\nİşlem yok.\n"
    cols = ["trades", "win_rate", "avg_win", "avg_loss", "expectancy", "expectancy_excess", "insufficient_data"]
    lines = [f"### {title}", "", "| Grup | İşlem | Kazanma | Ort. kazanç | Ort. kayıp | Expectancy | Endekse göre | Not |",
             "|---|---|---|---|---|---|---|---|"]
    for key, r in table[cols].iterrows():
        note = "yetersiz veri" if r["insufficient_data"] else ""
        lines.append(
            f"| {key} | {r['trades']:.0f} | {_fmt(r['win_rate'], '{:.1%}')} | {_fmt(r['avg_win'], '{:+.2%}')} "
            f"| {_fmt(r['avg_loss'], '{:+.2%}')} | {_fmt(r['expectancy'], '{:+.2%}')} "
            f"| {_fmt(r['expectancy_excess'], '{:+.2%}')} | {note} |"
        )
    return "\n".join(lines) + "\n"
