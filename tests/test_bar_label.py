"""bar 标签语义声明与归一的回归测试。

所有期望值直接来自 2026-07-24 的三源实测（futu / uSMART / longbridge），
不是从实现反推的 —— 见每个用例的注释。
"""

from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import pytest
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import BarData

from vnpy_gatewaykit.bar_label import (
    ENV_SWITCH,
    LABEL_SCHEMA_VERSION,
    RAW_LABEL_SCHEMA_VERSION,
    SOURCE_FUTU,
    SOURCE_LONGBRIDGE,
    SOURCE_USMART,
    AuctionPolicy,
    Confidence,
    LabelKind,
    LabelLedger,
    LabelSchemaConflict,
    SeriesKey,
    assert_trusted,
    is_trusted,
    label_spec,
    normalization_enabled,
    normalize_bars,
    relabel_stored_bars,
    start_label_on_grid,
    to_start_label,
)
from vnpy_gatewaykit.market_clock import market_tz

HK = Exchange.SEHK
US = Exchange.SMART

# 实测日：2026-07-24 是周五，港美股都是正常交易日。
HK_DAY = date(2026, 7, 24)
US_DAY = date(2026, 7, 23)


def hk(hour: int, minute: int, day: date = HK_DAY) -> datetime:
    return datetime.combine(day, time(hour, minute), tzinfo=market_tz(HK))


def us(hour: int, minute: int, day: date = US_DAY) -> datetime:
    return datetime.combine(day, time(hour, minute), tzinfo=market_tz(US))


def bar(stamp: datetime, exchange: Exchange, interval: Interval, volume: float = 1.0) -> BarData:
    symbol = "700" if exchange is HK else "AAPL"
    return BarData(
        gateway_name="TEST",
        symbol=symbol,
        exchange=exchange,
        datetime=stamp,
        interval=interval,
        volume=volume,
        open_price=1.0,
        high_price=1.0,
        low_price=1.0,
        close_price=1.0,
    )


# ---------------------------------------------------------------------------
# 1. 声明本身（实测事实的编码）
# ---------------------------------------------------------------------------
def test_futu_minute_declared_end_verified() -> None:
    spec = label_spec(SOURCE_FUTU, Interval.MINUTE)
    assert spec.kind is LabelKind.END
    assert spec.confidence is Confidence.VERIFIED
    assert spec.normalizable is True
    assert "320/332" in spec.evidence  # 移位互相关的决定性数字必须留在代码里


def test_futu_hour_declared_end_verified() -> None:
    spec = label_spec(SOURCE_FUTU, Interval.HOUR)
    assert spec.kind is LabelKind.END
    assert spec.confidence is Confidence.VERIFIED
    assert spec.normalizable is True
    # 午休残根是 60m 上最容易错的一根，证据里必须留着它的数字。
    assert "725268" in spec.evidence


def test_structural_confidence_is_also_normalizable() -> None:
    """STRUCTURAL 也允许改数据（只有 ASSUMED 不允许）—— 别把闸修错方向。"""
    from vnpy_gatewaykit.bar_label import LabelSpec

    assert LabelSpec(LabelKind.END, Confidence.STRUCTURAL, "x").normalizable is True
    assert LabelSpec(LabelKind.END, Confidence.ASSUMED, "x").normalizable is False


@pytest.mark.parametrize("source", [SOURCE_USMART, SOURCE_LONGBRIDGE])
def test_start_sources_are_not_normalizable(source: str) -> None:
    spec = label_spec(source, Interval.MINUTE)
    assert spec.kind is LabelKind.START
    assert spec.normalizable is False


@pytest.mark.parametrize("source", [SOURCE_FUTU, SOURCE_USMART])
def test_daily_is_session_date_never_normalizable(source: str) -> None:
    spec = label_spec(source, Interval.DAILY)
    assert spec.kind is LabelKind.SESSION_DATE
    assert spec.normalizable is False


@pytest.mark.parametrize("source", [SOURCE_FUTU, SOURCE_USMART])
def test_weekly_unmeasured_defaults_to_no_move(source: str) -> None:
    """周线锚点没实测过 ⇒ UNKNOWN ⇒ 永不平移（"默认不动"的落点）。"""
    spec = label_spec(source, Interval.WEEKLY)
    assert spec.kind is LabelKind.UNKNOWN
    assert spec.normalizable is False


def test_unregistered_source_defaults_to_no_move() -> None:
    spec = label_spec("some_new_broker", Interval.MINUTE)
    assert spec.kind is LabelKind.UNKNOWN
    assert spec.normalizable is False


def test_usmart_hour_is_declared_untrusted() -> None:
    """uSMART 60m 标签回来是 16:10 这种非栅格值，是窗口错位，平移救不了。"""
    assert is_trusted(SOURCE_USMART, Interval.HOUR) is False
    assert is_trusted(SOURCE_USMART, Interval.MINUTE) is True
    assert is_trusted(SOURCE_FUTU, Interval.HOUR) is True
    with pytest.raises(ValueError, match="16:10"):
        assert_trusted(SOURCE_USMART, Interval.HOUR)
    assert_trusted(SOURCE_USMART, Interval.MINUTE)  # 不抛


# ---------------------------------------------------------------------------
# 2. 港股分钟线：END -> START
# ---------------------------------------------------------------------------
# 左列 = futu 实测 time_key，右列 = uSMART/longbridge 实测 latestTime。
HK_MINUTE_CASES = [
    ((9, 31), (9, 30)),    # 连续盘首根
    ((10, 0), (9, 59)),
    ((11, 59), (11, 58)),
    ((12, 0), (11, 59)),   # 上午末根：futu 12:00 == uSMART 11:59
    ((13, 1), (13, 0)),    # 下午首根：futu 13:01 == uSMART 13:00
    ((16, 0), (15, 59)),   # 收盘末根：futu 16:00 == uSMART 15:59（成分含 CAS，见报告）
]


@pytest.mark.parametrize(("raw", "expected"), HK_MINUTE_CASES)
def test_hk_minute_end_to_start(raw: tuple[int, int], expected: tuple[int, int]) -> None:
    result = to_start_label(
        hk(*raw), source=SOURCE_FUTU, exchange=HK, interval=Interval.MINUTE, enabled=True
    )
    assert result.datetime == hk(*expected)
    assert result.changed is True
    assert result.note == "shifted:end->start"


def test_hk_minute_normalized_labels_equal_usmart_raw_labels() -> None:
    """决定性用例：归一后的 futu 标签集必须与 uSMART 原始标签集逐个相等。"""
    futu_labels = [hk(*raw) for raw, _ in HK_MINUTE_CASES]
    usmart_labels = [hk(*expected) for _, expected in HK_MINUTE_CASES]
    normalized = [
        to_start_label(
            stamp, source=SOURCE_FUTU, exchange=HK, interval=Interval.MINUTE, enabled=True
        ).datetime
        for stamp in futu_labels
    ]
    assert normalized == usmart_labels


def test_usmart_minute_left_alone() -> None:
    """uSMART 已经是 START，碰它就是把对的改错。"""
    for _, expected in HK_MINUTE_CASES:
        stamp = hk(*expected)
        result = to_start_label(
            stamp, source=SOURCE_USMART, exchange=HK, interval=Interval.MINUTE, enabled=True
        )
        assert result.datetime == stamp
        assert result.changed is False
        assert result.note.startswith("no-op:start")


# ---------------------------------------------------------------------------
# 3. 港股 60m：午休残根是裸减一个周期的反例
# ---------------------------------------------------------------------------
# futu 实测 60m 标签集 10:30/11:30/12:00/14:00/15:00/16:00。
HK_HOUR_CASES = [
    ((10, 30), (9, 30)),
    ((11, 30), (10, 30)),
    ((12, 0), (11, 30)),   # [11:30,12:00) 的 30 分钟残根 —— 裸减 60min 会得到 11:00
    ((14, 0), (13, 0)),
    ((15, 0), (14, 0)),
    ((16, 0), (15, 0)),
]


@pytest.mark.parametrize(("raw", "expected"), HK_HOUR_CASES)
def test_hk_hour_end_to_start_uses_session_grid(
    raw: tuple[int, int], expected: tuple[int, int]
) -> None:
    result = to_start_label(
        hk(*raw), source=SOURCE_FUTU, exchange=HK, interval=Interval.HOUR, enabled=True
    )
    assert result.datetime == hk(*expected)


def test_hk_hour_lunch_stub_differs_from_naive_minus_one_period() -> None:
    """帖子里的 `dt - timedelta(minutes=-1)` 式裸减在这里就是错的。"""
    raw = hk(12, 0)
    naive_shift = raw - timedelta(hours=1)
    grid = to_start_label(
        raw, source=SOURCE_FUTU, exchange=HK, interval=Interval.HOUR, enabled=True
    ).datetime
    assert naive_shift == hk(11, 0)
    assert grid == hk(11, 30)
    assert grid != naive_shift


def test_hk_hour_normalized_grid_has_no_duplicate_labels() -> None:
    normalized = {
        to_start_label(
            hk(*raw), source=SOURCE_FUTU, exchange=HK, interval=Interval.HOUR, enabled=True
        ).datetime
        for raw, _ in HK_HOUR_CASES
    }
    assert len(normalized) == len(HK_HOUR_CASES)


# ---------------------------------------------------------------------------
# 4. 港股竞价段
# ---------------------------------------------------------------------------
def test_hk_open_auction_bar_window_start_policy() -> None:
    """futu 单出的 09:30 竞价根覆盖 [09:00,09:30) ⇒ START = 09:00。"""
    result = to_start_label(
        hk(9, 30),
        source=SOURCE_FUTU,
        exchange=HK,
        interval=Interval.MINUTE,
        auction_policy=AuctionPolicy.WINDOW_START,
        enabled=True,
    )
    assert result.datetime == hk(9, 0)
    assert result.note == "auction:window-start"
    assert result.drop is False
    assert result.window is not None and result.window.name == "开市前竞价"


def test_hk_open_auction_window_start_avoids_collision_with_first_continuous_bar() -> None:
    first_continuous = to_start_label(
        hk(9, 31), source=SOURCE_FUTU, exchange=HK, interval=Interval.MINUTE, enabled=True
    ).datetime
    auction = to_start_label(
        hk(9, 30),
        source=SOURCE_FUTU,
        exchange=HK,
        interval=Interval.MINUTE,
        auction_policy=AuctionPolicy.WINDOW_START,
        enabled=True,
    ).datetime
    assert first_continuous == hk(9, 30)
    assert auction != first_continuous


def test_hk_open_auction_keep_policy_collides_on_purpose() -> None:
    auction = to_start_label(
        hk(9, 30),
        source=SOURCE_FUTU,
        exchange=HK,
        interval=Interval.MINUTE,
        auction_policy=AuctionPolicy.KEEP,
        enabled=True,
    )
    assert auction.datetime == hk(9, 30)
    assert auction.note == "auction:keep"


def test_hk_open_auction_drop_policy_flags_drop() -> None:
    auction = to_start_label(
        hk(9, 30),
        source=SOURCE_FUTU,
        exchange=HK,
        interval=Interval.MINUTE,
        auction_policy=AuctionPolicy.DROP,
        enabled=True,
    )
    assert auction.drop is True
    assert auction.note == "auction:drop"


def test_hk_cas_label_falls_in_closing_auction_window() -> None:
    """uSMART 单出的 16:00 CAS 根若来自 END 源，16:10 才是它的 END 标签。"""
    result = to_start_label(
        hk(16, 10), source=SOURCE_FUTU, exchange=HK, interval=Interval.MINUTE, enabled=True
    )
    assert result.window is not None and result.window.name == "收市竞价"
    assert result.datetime == hk(16, 0)


# ---------------------------------------------------------------------------
# 5. 美股：RTH + 盘前盘后
# ---------------------------------------------------------------------------
US_MINUTE_CASES = [
    ((4, 1), (4, 0)),      # 盘前首根（extended_time=True 才有）
    ((9, 30), (9, 29)),    # 盘前末根 —— 注意它不是 RTH 首根
    ((9, 31), (9, 30)),    # RTH 首根：futu 09:31 == uSMART 09:30
    ((16, 0), (15, 59)),   # RTH 末根：futu 16:00 == uSMART 15:59
    ((16, 1), (16, 0)),    # 盘后首根
    ((20, 0), (19, 59)),   # 盘后末根
]


@pytest.mark.parametrize(("raw", "expected"), US_MINUTE_CASES)
def test_us_minute_end_to_start(raw: tuple[int, int], expected: tuple[int, int]) -> None:
    result = to_start_label(
        us(*raw), source=SOURCE_FUTU, exchange=US, interval=Interval.MINUTE, enabled=True
    )
    assert result.datetime == us(*expected)


def test_us_rth_first_bar_is_not_confused_with_premarket_last_bar() -> None:
    """09:30 与 09:31 两个 END 标签必须落到不同窗口、不同 START。"""
    pre_last = to_start_label(
        us(9, 30), source=SOURCE_FUTU, exchange=US, interval=Interval.MINUTE, enabled=True
    )
    rth_first = to_start_label(
        us(9, 31), source=SOURCE_FUTU, exchange=US, interval=Interval.MINUTE, enabled=True
    )
    assert pre_last.window is not None and pre_last.window.name == "盘前"
    assert rth_first.window is not None and rth_first.window.name == "正常盘"
    assert pre_last.datetime == us(9, 29)
    assert rth_first.datetime == us(9, 30)


def test_us_hour_last_bar_is_a_thirty_minute_stub() -> None:
    """RTH 390 分钟 ⇒ 最后一根 60m 只有 [15:30,16:00) 的 30 分钟。"""
    assert to_start_label(
        us(10, 30), source=SOURCE_FUTU, exchange=US, interval=Interval.HOUR, enabled=True
    ).datetime == us(9, 30)
    assert to_start_label(
        us(16, 0), source=SOURCE_FUTU, exchange=US, interval=Interval.HOUR, enabled=True
    ).datetime == us(15, 30)


def test_label_outside_every_session_window_is_left_alone() -> None:
    """美股 02:00 ET 不在任何窗口里（04:00 才开盘前）⇒ 不猜，原样放行。"""
    stamp = us(2, 0)
    result = to_start_label(
        stamp, source=SOURCE_FUTU, exchange=US, interval=Interval.MINUTE, enabled=True
    )
    assert result.datetime == stamp
    assert result.note == "unchanged:outside-all-windows"


# ---------------------------------------------------------------------------
# 6. 日线 / 周线：过度修正防护
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("interval", [Interval.DAILY, Interval.WEEKLY])
@pytest.mark.parametrize("exchange", [HK, US])
def test_daily_and_weekly_never_shift(interval: Interval, exchange: Exchange) -> None:
    stamp = datetime.combine(HK_DAY, time(0, 0), tzinfo=market_tz(exchange))
    result = to_start_label(
        stamp, source=SOURCE_FUTU, exchange=exchange, interval=interval, enabled=True
    )
    assert result.datetime == stamp
    assert result.changed is False


def test_daily_midnight_is_not_dragged_into_the_session_grid() -> None:
    """00:00 若被当成日内标签会掉进"不在任何窗口"分支甚至被平移 —— 必须先被 kind 拦住。"""
    stamp = datetime.combine(HK_DAY, time(0, 0), tzinfo=market_tz(HK))
    result = to_start_label(
        stamp, source=SOURCE_FUTU, exchange=HK, interval=Interval.DAILY, enabled=True
    )
    assert result.note == "no-op:session_date/verified"


# ---------------------------------------------------------------------------
# 7. DST
# ---------------------------------------------------------------------------
def test_us_dst_boundary_days_keep_correct_utc_offset() -> None:
    """2026 美国 DST：3/8 春进、11/1 秋退。前后交易日的偏移必须各自正确。"""
    est_day = date(2026, 3, 6)   # 周五，EST，UTC-5
    edt_day = date(2026, 3, 9)   # 周一，EDT，UTC-4
    back_day = date(2026, 11, 2)  # 周一，秋退之后，EST，UTC-5

    for day, offset_hours in ((est_day, -5), (edt_day, -4), (back_day, -5)):
        raw = us(9, 31, day)
        assert raw.utcoffset() == timedelta(hours=offset_hours)
        result = to_start_label(
            raw, source=SOURCE_FUTU, exchange=US, interval=Interval.MINUTE, enabled=True
        )
        assert result.datetime == us(9, 30, day)
        assert result.datetime.utcoffset() == timedelta(hours=offset_hours)


def test_us_dst_shift_is_exactly_sixty_real_seconds() -> None:
    """UTC 口径：一根分钟 bar 是 60 个真实秒。"""
    raw = us(9, 31, date(2026, 3, 9))
    result = to_start_label(
        raw, source=SOURCE_FUTU, exchange=US, interval=Interval.MINUTE, enabled=True
    )
    assert (raw - result.datetime).total_seconds() == 60.0


def test_utc_input_is_converted_to_market_local() -> None:
    raw = us(9, 31).astimezone(timezone.utc)  # noqa: UP017 — 3.10 兼容，同 sessions.py
    result = to_start_label(
        raw, source=SOURCE_FUTU, exchange=US, interval=Interval.MINUTE, enabled=True
    )
    assert result.datetime == us(9, 30)


def test_naive_datetime_rejected() -> None:
    with pytest.raises(ValueError, match="tz-aware"):
        to_start_label(
            datetime(2026, 7, 24, 9, 31),
            source=SOURCE_FUTU,
            exchange=HK,
            interval=Interval.MINUTE,
            enabled=True,
        )


# ---------------------------------------------------------------------------
# 8. 全局开关
# ---------------------------------------------------------------------------
def test_env_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_SWITCH, "0")
    assert normalization_enabled() is False
    result = to_start_label(
        hk(9, 31), source=SOURCE_FUTU, exchange=HK, interval=Interval.MINUTE
    )
    assert result.datetime == hk(9, 31)
    assert result.note == f"disabled:{ENV_SWITCH}"


def test_env_default_is_off_until_history_is_migrated(monkeypatch: pytest.MonkeyPatch) -> None:
    """默认必须关闭。

    归一本身是对的（futu 日内 K 是 END 标签，vnpy/uSMART/longbridge 是 START），
    但打开它会让新写入的 bar 与已落库的历史 bar 相差一个周期 —— 同一根 K 线在库里
    出现两个时间戳，静默发生、回测读到的序列在迁移分界点错位一格。
    历史数据迁移完成前，安全默认是不动。
    """
    monkeypatch.delenv(ENV_SWITCH, raising=False)
    assert normalization_enabled() is False


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off", ""])
def test_env_falsey_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(ENV_SWITCH, value)
    assert normalization_enabled() is False


def test_explicit_enabled_beats_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_SWITCH, "0")
    result = to_start_label(
        hk(9, 31), source=SOURCE_FUTU, exchange=HK, interval=Interval.MINUTE, enabled=True
    )
    assert result.datetime == hk(9, 30)


# ---------------------------------------------------------------------------
# 9. 整段归一
# ---------------------------------------------------------------------------
def _hk_futu_minute_day() -> list[BarData]:
    """一个缩略的港股 futu 分钟序列：竞价根 + 两段连续盘的首末根。"""
    stamps = [(9, 30), (9, 31), (11, 59), (12, 0), (13, 1), (15, 59), (16, 0)]
    return [bar(hk(*s), HK, Interval.MINUTE) for s in stamps]


def test_normalize_bars_report_counts() -> None:
    bars, report = normalize_bars(_hk_futu_minute_day(), source=SOURCE_FUTU, enabled=True)
    assert report.total == 7
    assert report.auction == 1
    assert report.dropped == 0
    assert report.shifted == 7  # 竞价根 09:30 -> 09:00 也变了
    assert report.unchanged == 0
    assert report.collisions == ()
    assert [b.datetime for b in bars] == [
        hk(9, 0), hk(9, 30), hk(11, 58), hk(11, 59), hk(13, 0), hk(15, 58), hk(15, 59)
    ]
    assert "shifted:end->start" in report.summary() or report.summary().startswith("source=")


def test_normalize_bars_does_not_mutate_input() -> None:
    original = _hk_futu_minute_day()
    stamps_before = [b.datetime for b in original]
    normalize_bars(original, source=SOURCE_FUTU, enabled=True)
    assert [b.datetime for b in original] == stamps_before


def test_normalize_bars_preserves_ohlcv_and_identity() -> None:
    src = bar(hk(9, 31), HK, Interval.MINUTE, volume=1061900.0)
    src.turnover = 4.6e8
    out, _ = normalize_bars([src], source=SOURCE_FUTU, enabled=True)
    assert out[0] is not src
    assert out[0].datetime == hk(9, 30)
    assert out[0].volume == 1061900.0
    assert out[0].turnover == 4.6e8
    assert out[0].vt_symbol == src.vt_symbol
    assert out[0].gateway_name == src.gateway_name
    assert out[0].interval is Interval.MINUTE


def test_normalize_bars_keep_policy_reports_collision() -> None:
    """KEEP 下竞价根与首根连续 bar 撞在 09:30 —— 报告必须点名。"""
    _, report = normalize_bars(
        _hk_futu_minute_day(),
        source=SOURCE_FUTU,
        auction_policy=AuctionPolicy.KEEP,
        enabled=True,
    )
    assert report.collisions == ((hk(9, 30), 2),)


def test_normalize_bars_drop_policy_removes_auction_bar() -> None:
    bars, report = normalize_bars(
        _hk_futu_minute_day(),
        source=SOURCE_FUTU,
        auction_policy=AuctionPolicy.DROP,
        enabled=True,
    )
    assert report.dropped == 1
    assert len(bars) == 6
    assert hk(9, 0) not in {b.datetime for b in bars}


def test_normalize_bars_is_noop_for_usmart() -> None:
    src = [bar(hk(9, 30), HK, Interval.MINUTE), bar(hk(11, 59), HK, Interval.MINUTE)]
    out, report = normalize_bars(src, source=SOURCE_USMART, enabled=True)
    assert [b.datetime for b in out] == [b.datetime for b in src]
    assert report.shifted == 0
    assert report.unchanged == 2


def test_normalize_bars_handles_interval_none() -> None:
    src = bar(hk(9, 31), HK, Interval.MINUTE)
    src.interval = None
    out, report = normalize_bars([src], source=SOURCE_FUTU, enabled=True)
    assert out[0].datetime == hk(9, 31)
    assert report.unchanged == 1
    assert report.notes["no-op:interval-is-none"] == 1


def test_normalize_bars_disabled_is_pure_passthrough() -> None:
    src = _hk_futu_minute_day()
    out, report = normalize_bars(src, source=SOURCE_FUTU, enabled=False)
    assert [b.datetime for b in out] == [b.datetime for b in src]
    assert report.shifted == 0
    assert report.unchanged == len(src)


def test_double_normalization_shifts_twice_so_the_ledger_is_mandatory() -> None:
    """归一**不是**幂等的：值上看不出一根 bar 是否已经归过一次。

    这正是台账存在的理由 —— 防重复只能靠版本声明，不能靠"再跑一遍应该没事"。
    """
    once, _ = normalize_bars(
        [bar(hk(10, 0), HK, Interval.MINUTE)], source=SOURCE_FUTU, enabled=True
    )
    twice, _ = normalize_bars(once, source=SOURCE_FUTU, enabled=True)
    assert once[0].datetime == hk(9, 59)
    assert twice[0].datetime == hk(9, 58)

    # 连续盘首根更糟：第二次归一会把它当成竞价根，一次性挪 31 分钟。
    first_once, _ = normalize_bars(
        [bar(hk(9, 31), HK, Interval.MINUTE)], source=SOURCE_FUTU, enabled=True
    )
    first_twice, _ = normalize_bars(first_once, source=SOURCE_FUTU, enabled=True)
    assert first_once[0].datetime == hk(9, 30)
    assert first_twice[0].datetime == hk(9, 0)


# ---------------------------------------------------------------------------
# 10. 离线迁移
# ---------------------------------------------------------------------------
def test_relabel_v1_to_v2() -> None:
    out, report = relabel_stored_bars(
        [bar(hk(13, 1), HK, Interval.MINUTE)],
        source=SOURCE_FUTU,
        from_version=RAW_LABEL_SCHEMA_VERSION,
        to_version=LABEL_SCHEMA_VERSION,
    )
    assert out[0].datetime == hk(13, 0)
    assert report.shifted == 1


def test_relabel_same_version_is_noop() -> None:
    src = [bar(hk(13, 1), HK, Interval.MINUTE)]
    out, report = relabel_stored_bars(
        src, source=SOURCE_FUTU, from_version=2, to_version=2
    )
    assert out[0].datetime == hk(13, 1)
    assert report.unchanged == 1


def test_relabel_unsupported_direction_raises() -> None:
    with pytest.raises(ValueError, match="不支持的标签口径迁移"):
        relabel_stored_bars(
            [bar(hk(13, 1), HK, Interval.MINUTE)],
            source=SOURCE_FUTU,
            from_version=LABEL_SCHEMA_VERSION,
            to_version=RAW_LABEL_SCHEMA_VERSION,
        )


def test_relabel_ignores_env_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    """迁移是显式动作，不该被"新数据先别归一"的现场开关顺手关掉。"""
    monkeypatch.setenv(ENV_SWITCH, "0")
    out, _ = relabel_stored_bars(
        [bar(hk(13, 1), HK, Interval.MINUTE)],
        source=SOURCE_FUTU,
        from_version=RAW_LABEL_SCHEMA_VERSION,
    )
    assert out[0].datetime == hk(13, 0)


# ---------------------------------------------------------------------------
# 11. SeriesKey
# ---------------------------------------------------------------------------
def test_series_key_roundtrip() -> None:
    key = SeriesKey(SOURCE_FUTU, "700", HK, Interval.MINUTE)
    assert key.text == "futu|700.SEHK|1m"
    assert SeriesKey.parse(key.text) == key


def test_series_key_of_bar() -> None:
    key = SeriesKey.of(bar(hk(9, 31), HK, Interval.MINUTE), SOURCE_FUTU)
    assert key == SeriesKey(SOURCE_FUTU, "700", HK, Interval.MINUTE)


def test_series_key_of_bar_without_interval_raises() -> None:
    src = bar(hk(9, 31), HK, Interval.MINUTE)
    src.interval = None
    with pytest.raises(ValueError, match="没有 interval"):
        SeriesKey.of(src, SOURCE_FUTU)


def test_series_key_parse_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="非法的 SeriesKey"):
        SeriesKey.parse("garbage")


# ---------------------------------------------------------------------------
# 12. 台账
# ---------------------------------------------------------------------------
@pytest.fixture()
def ledger_path(tmp_path: Path) -> Path:
    return tmp_path / "labels" / "bar_label_ledger.json"


def test_ledger_declare_and_read_back(ledger_path: Path) -> None:
    key = SeriesKey(SOURCE_FUTU, "700", HK, Interval.MINUTE)
    ledger = LabelLedger(ledger_path)
    assert ledger.version_of(key) is None
    ledger.declare(key, LABEL_SCHEMA_VERSION)
    assert ledger.version_of(key) == LABEL_SCHEMA_VERSION
    assert LabelLedger(ledger_path).version_of(key) == LABEL_SCHEMA_VERSION
    assert json.loads(ledger_path.read_text(encoding="utf-8")) == {
        "futu|700.SEHK|1m": LABEL_SCHEMA_VERSION
    }


def test_ledger_blocks_mixing_old_and_new_schema(ledger_path: Path) -> None:
    key = SeriesKey(SOURCE_FUTU, "700", HK, Interval.MINUTE)
    ledger = LabelLedger(ledger_path)
    ledger.bootstrap([key], RAW_LABEL_SCHEMA_VERSION)
    with pytest.raises(LabelSchemaConflict, match="v1"):
        ledger.assert_compatible(key, LABEL_SCHEMA_VERSION)
    ledger.assert_compatible(key, RAW_LABEL_SCHEMA_VERSION)  # 同口径放行


def test_ledger_declare_conflict_needs_overwrite(ledger_path: Path) -> None:
    key = SeriesKey(SOURCE_FUTU, "700", HK, Interval.MINUTE)
    ledger = LabelLedger(ledger_path)
    ledger.declare(key, RAW_LABEL_SCHEMA_VERSION)
    with pytest.raises(LabelSchemaConflict):
        ledger.declare(key, LABEL_SCHEMA_VERSION)
    ledger.declare(key, LABEL_SCHEMA_VERSION, overwrite=True)
    assert ledger.version_of(key) == LABEL_SCHEMA_VERSION


def test_ledger_unknown_series_passes_by_default_but_can_be_required(ledger_path: Path) -> None:
    key = SeriesKey(SOURCE_FUTU, "AAPL", US, Interval.MINUTE)
    ledger = LabelLedger(ledger_path)
    ledger.assert_compatible(key, LABEL_SCHEMA_VERSION)  # 未登记：默认放行
    with pytest.raises(LabelSchemaConflict, match="未在标签口径台账中登记"):
        ledger.assert_compatible(key, LABEL_SCHEMA_VERSION, require_declared=True)


def test_ledger_bootstrap_conflict_needs_overwrite(ledger_path: Path) -> None:
    key = SeriesKey(SOURCE_FUTU, "700", HK, Interval.MINUTE)
    ledger = LabelLedger(ledger_path)
    ledger.bootstrap([key], RAW_LABEL_SCHEMA_VERSION)
    with pytest.raises(LabelSchemaConflict):
        ledger.bootstrap([key], LABEL_SCHEMA_VERSION)
    ledger.bootstrap([key], LABEL_SCHEMA_VERSION, overwrite=True)
    assert ledger.version_of(key) == LABEL_SCHEMA_VERSION


def test_ledger_keys_listing(ledger_path: Path) -> None:
    a = SeriesKey(SOURCE_FUTU, "700", HK, Interval.MINUTE)
    b = SeriesKey(SOURCE_USMART, "AAPL", US, Interval.DAILY)
    ledger = LabelLedger(ledger_path)
    ledger.bootstrap([a, b], LABEL_SCHEMA_VERSION)
    assert set(ledger.keys()) == {a, b}


def test_ledger_corrupt_file_raises_instead_of_silently_resetting(ledger_path: Path) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="已损坏"):
        LabelLedger(ledger_path)


def test_ledger_rejects_non_integer_version(ledger_path: Path) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text('{"futu|700.SEHK|1m": "two"}', encoding="utf-8")
    with pytest.raises(ValueError, match="版本号非整数"):
        LabelLedger(ledger_path)


def test_ledger_rejects_non_object_top_level(ledger_path: Path) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="顶层应是对象"):
        LabelLedger(ledger_path)


def test_ledger_write_leaves_no_temp_files(ledger_path: Path) -> None:
    key = SeriesKey(SOURCE_FUTU, "700", HK, Interval.MINUTE)
    LabelLedger(ledger_path).declare(key, LABEL_SCHEMA_VERSION)
    assert sorted(p.name for p in ledger_path.parent.iterdir()) == ["bar_label_ledger.json"]


# ---------------------------------------------------------------------------
# start_label_on_grid：周期由调用方给的栅格原语（CSV 批量导入走这条）
# ---------------------------------------------------------------------------
def test_grid_primitive_matches_to_start_label_on_the_registered_path() -> None:
    """原语与注册表入口共用同一份栅格数学，不允许有第二种答案。"""
    tz = market_tz(HK)
    for label in ("10:30", "11:30", "12:00", "14:00", "16:00"):
        moment = datetime.strptime(f"2026-07-24 {label}", "%Y-%m-%d %H:%M").replace(
            tzinfo=tz
        )
        # 显式 enabled=True：这里比的是两条路径的【栅格数学】是否一致，
        # 与全局开关无关（开关默认关闭，见 test_env_default_is_off_...）。
        registered = to_start_label(
            moment, source=SOURCE_FUTU, exchange=HK, interval=Interval.HOUR,
            enabled=True,
        )
        primitive = start_label_on_grid(
            moment, exchange=HK, period=timedelta(hours=1)
        )
        assert primitive.datetime == registered.datetime


def test_grid_primitive_handles_a_span_that_is_not_the_interval() -> None:
    """5 分钟 vendor 导出以 Interval.MINUTE 落库时，栅格步长必须是 5 分钟。

    ``to_start_label`` 走不了这条：它的周期由 ``Interval`` 定死。
    """
    tz = market_tz(HK)
    moment = datetime(2026, 7, 24, 10, 5, tzinfo=tz)
    result = start_label_on_grid(moment, exchange=HK, period=timedelta(minutes=5))
    assert result.datetime == datetime(2026, 7, 24, 10, 0, tzinfo=tz)


def test_grid_primitive_snaps_the_lunch_truncated_hour_to_1130() -> None:
    """裸减一个周期会得到 11:00（落在上一根里）；栅格重建得 11:30。"""
    tz = market_tz(HK)
    moment = datetime(2026, 7, 24, 12, 0, tzinfo=tz)
    result = start_label_on_grid(moment, exchange=HK, period=timedelta(hours=1))
    assert result.datetime == datetime(2026, 7, 24, 11, 30, tzinfo=tz)
    assert result.note == "shifted:end->start"


def test_grid_primitive_snaps_the_us_close_stub_to_1530() -> None:
    tz = market_tz(US)
    moment = datetime(2026, 7, 24, 16, 0, tzinfo=tz)
    result = start_label_on_grid(moment, exchange=US, period=timedelta(hours=1))
    assert result.datetime == datetime(2026, 7, 24, 15, 30, tzinfo=tz)


def test_grid_primitive_reports_stamps_no_window_contains() -> None:
    """午休时刻不属于任何窗口 —— 返回原值并标明，不猜。"""
    tz = market_tz(HK)
    moment = datetime(2026, 7, 24, 12, 30, tzinfo=tz)
    result = start_label_on_grid(moment, exchange=HK, period=timedelta(hours=1))
    assert result.note == "unchanged:outside-all-windows"
    assert result.datetime == moment


def test_grid_primitive_rejects_naive_and_non_positive_period() -> None:
    tz = market_tz(HK)
    with pytest.raises(ValueError, match="tz-aware"):
        # 裸墙钟时刻正是本用例要拒的输入，naive 是刻意的。
        start_label_on_grid(
            datetime(2026, 7, 24, 12, 0),  # noqa: DTZ001
            exchange=HK,
            period=timedelta(hours=1),
        )
    with pytest.raises(ValueError, match="必须为正"):
        start_label_on_grid(
            datetime(2026, 7, 24, 12, 0, tzinfo=tz), exchange=HK, period=timedelta(0)
        )


def test_grid_primitive_honours_auction_policy() -> None:
    tz = market_tz(HK)
    moment = datetime(2026, 7, 24, 9, 30, tzinfo=tz)
    assert start_label_on_grid(
        moment, exchange=HK, period=timedelta(minutes=1)
    ).datetime == datetime(2026, 7, 24, 9, 0, tzinfo=tz)
    assert start_label_on_grid(
        moment,
        exchange=HK,
        period=timedelta(minutes=1),
        auction_policy=AuctionPolicy.DROP,
    ).drop is True
