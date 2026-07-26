"""stored_label_version —— 把"这段序列现在是什么口径"变成可调用的一行。

LabelLedger 立在那里靠的是写库方在写之前算出"我要写的这批 bar 是 v1 还是 v2"。
在此之前每个写库方都得自己复述一遍 to_start_label 的分支条件(开关 + 是否
normalizable + 有没有栅格),复述错了台账就记错版本 —— 一个记错版本的台账比没有
台账更糟。所以判定放在标签知识所在的这个模块里，写库方只调用。
"""

from __future__ import annotations

import pytest

from vnpy.trader.constant import Interval

from vnpy_gatewaykit.bar_label import (
    LABEL_SCHEMA_VERSION,
    RAW_LABEL_SCHEMA_VERSION,
    SOURCE_FUTU,
    SOURCE_LONGBRIDGE,
    SOURCE_USMART,
    stored_label_version,
)


@pytest.mark.parametrize("interval", [Interval.MINUTE, Interval.HOUR])
def test_futu_intraday_follows_the_switch(interval: Interval) -> None:
    """futu 日内是 END 语义,归一会真的平移 —— 两个开关状态是两个口径。"""
    assert stored_label_version(SOURCE_FUTU, interval, enabled=False) == RAW_LABEL_SCHEMA_VERSION
    assert stored_label_version(SOURCE_FUTU, interval, enabled=True) == LABEL_SCHEMA_VERSION


def test_futu_daily_never_shifts_so_it_is_always_current() -> None:
    """日线标签是交易日的日期标识,永不平移;开关开着也一样。
    把它记成 v1 会在开关翻转那天造出一个假冲突。"""
    assert stored_label_version(SOURCE_FUTU, Interval.DAILY, enabled=False) == LABEL_SCHEMA_VERSION
    assert stored_label_version(SOURCE_FUTU, Interval.DAILY, enabled=True) == LABEL_SCHEMA_VERSION


@pytest.mark.parametrize("source", [SOURCE_USMART, SOURCE_LONGBRIDGE])
def test_native_start_sources_are_always_current(source: str) -> None:
    """uSMART / longbridge 分钟线本来就是 START,归一是 no-op。"""
    assert stored_label_version(source, Interval.MINUTE, enabled=False) == LABEL_SCHEMA_VERSION
    assert stored_label_version(source, Interval.MINUTE, enabled=True) == LABEL_SCHEMA_VERSION


def test_unknown_source_is_never_shifted() -> None:
    """未登记的源一律"默认不动"(label_spec 返回 UNKNOWN),所以它写出来的东西
    在两个口径下完全一样。"""
    assert stored_label_version("nasdaq-csv", Interval.MINUTE, enabled=True) == (
        LABEL_SCHEMA_VERSION
    )


def test_weekly_has_no_grid_so_it_is_never_shifted() -> None:
    assert stored_label_version(SOURCE_FUTU, Interval.WEEKLY, enabled=True) == (
        LABEL_SCHEMA_VERSION
    )


def test_default_enabled_reads_the_env_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VNPY_BAR_LABEL_NORMALIZE", "0")
    assert stored_label_version(SOURCE_FUTU, Interval.MINUTE) == RAW_LABEL_SCHEMA_VERSION
    monkeypatch.setenv("VNPY_BAR_LABEL_NORMALIZE", "1")
    assert stored_label_version(SOURCE_FUTU, Interval.MINUTE) == LABEL_SCHEMA_VERSION
