"""Bar 时间戳标签语义（START / END）的声明与归一。

一根 K 线的 datetime 到底指它覆盖区间的**开始**还是**结束**，是数据源自己的
约定，wire 上没有任何字段说明它。vnpy 核心是 START 语义
(``vnpy.trader.utility.BarGenerator.update_tick`` 用该分钟**第一个** tick 的
时间建 bar)，所以任何 END 语义的源在落库前必须归一，否则同一标的同一时段用两
个网关灌库会得到两根差一个周期的 bar。

实测事实（2026-07-24 HK.00700 / US.AAPL，与 longbridge 三方交叉验证）：

* **futu = END**：``request_history_kline`` 的 ``time_key`` 是收尾时刻。
  港股上午最后一根标 12:00、下午第一根标 13:01；美股 390 根 RTH 分钟线标
  09:31..16:00。
* **uSMART = START**：``kline`` 的 ``latestTime`` 是起始时刻。港股上午最后一根
  标 11:59、下午第一根标 13:00；美股标 09:30..16:00（391 根，含 CAS）。
* **longbridge 独立仲裁 = START**，与 uSMART 逐根吻合（成交量完全相同）。
* 移位互相关（决定性）：``futu_label = usmart_label + 1 期`` 时
  volume-match 320/332、OHLC-exact 287/332；其余移位量 OHLC-exact 均 ≤26。

所以本模块的默认动作是：**把 futu 的日内 K 线标签归一到 START**，其余源不动。

三条容易踩坑的边界，本模块逐条处理：

1. **日线/周线不是同一种标签语义。** futu 日线 ``time_key`` 是
   ``2026-07-24 00:00:00``、uSMART 是 ``20260724000000000`` —— 两家一致，而且
   00:00 既不是 session 开始(09:30)也不是结束(16:00)：它是**交易日本身的日期
   标识**，不是一个瞬时。所以日线/周线归类为 :attr:`LabelKind.SESSION_DATE`，
   **永不平移**；把日线"修正"到 09:30 会打断 datamanager / QuestDB / 图表对
   "日线主键 = 日期"的假设。周线的锚点（周一还是周五）未实测 → 标记
   :attr:`LabelKind.UNKNOWN`，同样永不平移。

2. **裸减一个周期在 session 边界上是错的。** 港股 60m 的 futu 标签是
   ``10:30/11:30/12:00/14:00/15:00/16:00``：标 12:00 的那根覆盖
   ``[11:30, 12:00)``（午休截断出的 30 分钟残根），裸减 60 分钟会得到 11:00 ——
   落在上一根的窗口里。本模块不做裸减，而是**按 session 窗口重建栅格**：
   在包含该标签的窗口内取"小于该标签的最大栅格点"，于是 12:00 → 11:30。

3. **竞价段是 bar 组成方式的差异，不是标签差异，平移修不干净。** 港股开盘
   futu 单出一根 09:30 竞价 bar(V=545400) + 09:31 起连续；uSMART/longbridge 把
   "竞价+首分钟"并成一根 09:30(V=1646424)。港股收盘反过来：uSMART 单出 16:00
   CAS(V=1277300)，futu 并进 16:00(V=1334300 = 57000 + 1277300，精确相等)。
   平移只能让**标签**对齐，OHLCV 在开盘首根/收盘末根仍然对不上。本模块因此
   ① 用 :class:`AuctionPolicy` 显式声明竞价 bar 怎么处理，
   ② 在 :class:`NormalizeReport` 里把竞价根与标签碰撞单独计数，
   ③ 提供 :class:`LabelLedger` 让"已落库的旧口径"与"归一后的新口径"不会混。

DST：所有栅格点都由 ``datetime.combine(day, session_time, tzinfo=market_tz)``
生成，再在 **UTC 上**做加减（一根分钟 bar 是 60 个真实秒，不是 60 个墙钟秒）。
美股 04:00/09:30/16:00/20:00 ET 没有一个落在 02:00–03:00 ET 的 DST 切换小时里
（切换发生在周日、闭市），所以本模块永远不会算到一个不存在或二义的本地时刻。
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from pathlib import Path

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import BarData

from .market_clock import market_tz
from .sessions import Session, SessionKind, sessions_for

__all__ = [
    "ENV_SWITCH",
    "LABEL_SCHEMA_VERSION",
    "RAW_LABEL_SCHEMA_VERSION",
    "SOURCE_FUTU",
    "SOURCE_LABELS",
    "SOURCE_LONGBRIDGE",
    "SOURCE_REPLAY",
    "SOURCE_USMART",
    "AuctionPolicy",
    "Confidence",
    "LabelKind",
    "LabelLedger",
    "LabelResult",
    "LabelSchemaConflict",
    "LabelSpec",
    "NormalizeReport",
    "SeriesKey",
    "assert_trusted",
    "is_trusted",
    "label_spec",
    "normalization_enabled",
    "normalize_bars",
    "relabel_stored_bars",
    "start_label_on_grid",
    "to_start_label",
]


# ---------------------------------------------------------------------------
# 版本 / 开关
# ---------------------------------------------------------------------------
#: 未经归一、直接透传数据源标签的落库口径（本模块上线前所有已落库数据）。
RAW_LABEL_SCHEMA_VERSION: int = 1

#: 归一到 vnpy 原生 START 语义后的落库口径。
LABEL_SCHEMA_VERSION: int = 2

#: 全局急停开关。设为 0/false/no/off 时本模块一律不动任何标签，
#: 用于"怀疑归一本身出问题、先回到旧口径"的现场处置。
ENV_SWITCH: str = "VNPY_BAR_LABEL_NORMALIZE"

_FALSEY: frozenset[str] = frozenset({"0", "false", "no", "off", ""})


def normalization_enabled() -> bool:
    """读取全局开关。

    **默认关闭。** 归一本身是对的（futu 日内 K 用 END 标签，vnpy/uSMART/longbridge
    用 START，已由三方对账实测确认），但打开它会让新写入的 bar 与【已经落库的
    历史 bar】相差一个周期 —— 同一根 K 线在库里出现两个时间戳，而且是静默发生的。

    在完成历史数据迁移（重灌或整体平移）之前，默认开启等于制造数据污染：
    回测读到的序列会在迁移分界点上错位一格，且没有任何报错。

    迁移完成后设 VNPY_BAR_LABEL_NORMALIZE=1 打开，或在调用点显式传 enabled=True
    （单元测试就是这么做的，所以标签语义仍被测试覆盖）。
    """
    return os.environ.get(ENV_SWITCH, "0").strip().lower() not in _FALSEY


def _resolve_enabled(enabled: bool | None) -> bool:
    return normalization_enabled() if enabled is None else enabled


# ---------------------------------------------------------------------------
# 标签语义声明
# ---------------------------------------------------------------------------
class LabelKind(Enum):
    """一个 (源, 周期) 的 datetime 到底指什么。"""

    #: bar 覆盖区间的开始时刻。vnpy 原生语义，归一的目标。
    START = "start"
    #: bar 覆盖区间的结束时刻。需要归一。
    END = "end"
    #: 交易日 / 交易周本身的日期标识（日线/周线），不是瞬时，永不平移。
    SESSION_DATE = "session_date"
    #: 未实测确认。永不平移 —— 这是"可配置 + 默认不动"的落点。
    UNKNOWN = "unknown"


class Confidence(Enum):
    """这条声明的证据强度。只有 VERIFIED / STRUCTURAL 才允许改动数据。"""

    #: 有跨源逐根比对的实测证据。
    VERIFIED = "verified"
    #: 由标签栅格的结构本身推出（例如整个序列里不存在 09:30 这个标签，
    #: 只可能是 END 语义），无逐根比对。
    STRUCTURAL = "structural"
    #: 只是猜测 / 照抄文档。**不允许**据此改动数据。
    ASSUMED = "assumed"


class AuctionPolicy(Enum):
    """竞价窗口内的 bar 怎么处理（开盘竞价 / 收市竞价）。"""

    #: 平移到竞价窗口的开始（港股 09:30 竞价根 → 09:00）。无损、且不会与
    #: 归一后的首根连续 bar(09:30) 撞标签。默认。
    WINDOW_START = "window_start"
    #: 原样保留。会与归一后的首根连续 bar 撞标签，碰撞会被报告出来。
    KEEP = "keep"
    #: 标记为丢弃，由 :func:`normalize_bars` 真正剔除（会丢掉竞价成交量）。
    DROP = "drop"


@dataclass(frozen=True, slots=True)
class LabelSpec:
    """某个 (源, 周期) 的标签语义声明 + 证据。"""

    kind: LabelKind
    confidence: Confidence
    evidence: str
    #: False = 该 (源, 周期) 的数据本身不可信，不该落库（不是标签问题，平移救不了）。
    trusted: bool = True

    @property
    def normalizable(self) -> bool:
        """是否允许据此改动数据。"""
        return self.kind is LabelKind.END and self.confidence in (
            Confidence.VERIFIED,
            Confidence.STRUCTURAL,
        )


SOURCE_FUTU: str = "futu"
SOURCE_USMART: str = "usmart"
SOURCE_LONGBRIDGE: str = "longbridge"
SOURCE_REPLAY: str = "replay"

_UNKNOWN_SPEC = LabelSpec(
    LabelKind.UNKNOWN,
    Confidence.ASSUMED,
    "未实测该 (源, 周期) 的标签语义；按'默认不动'处理",
)

_FUTU_INTRADAY_EVIDENCE = (
    "2026-07-24 HK.00700/US.AAPL 实测：futu time_key 与 uSMART latestTime 的移位"
    "互相关在 +1 期取得 volume-match 320/332、OHLC-exact 287/332（其余移位量 "
    "OHLC-exact ≤26）；longbridge 独立仲裁与 uSMART 逐根吻合。港股 futu 上午末根"
    "标 12:00、下午首根标 13:01 ⇒ END。"
)

_FUTU_HOUR_EVIDENCE = (
    "2026-07-24 HK.00700 futu 60m 标签集 = 10:30/11:30/12:00/14:00/15:00/16:00；"
    "归一后 = 09:30/10:30/11:30/13:00/14:00/15:00，与 longbridge(独立第三方) 同日"
    "60m 的 START 栅格逐根重合。含午休残根这一关键点：futu 12:00 → 11:30，"
    "longbridge 11:30 V=725268 vs futu 725100(差 0.02%)；裸减 60 分钟会得到 11:00，"
    "而 longbridge 在 11:00 根本没有 bar。"
)

_DAILY_EVIDENCE = (
    "2026-07-24 实测：futu 日线 time_key='2026-07-24 00:00:00'、uSMART "
    "latestTime=20260724000000000，两家一致且都不是 session 起止时刻 —— 日线标签"
    "是交易日的日期标识，不是瞬时，不可平移。"
)

_USMART_MINUTE_EVIDENCE = (
    "2026-07-24 hk00700/usAAPL 实测：上午末根 11:59、下午首根 13:00、美股首根 "
    "09:30；与 longbridge(独立第三方) 逐根吻合，含成交量 ⇒ START，已经是 vnpy 口径。"
)

_USMART_HOUR_EVIDENCE = (
    "2026-07-24 实测：uSMART type=6(60m) 港股标签回来是 "
    "10:30/11:30/13:30/14:30/15:30/16:10 —— 16:10 不在任何合理栅格上；5m/30m 用 "
    "futu 1m 重建验证窗口整体偏移 +1 分钟（偏移窗口命中 57/66，自然栅格 4/66）。"
    "这是**聚合窗口本身错位**，不是标签问题，平移救不了 ⇒ trusted=False，禁止落库，"
    "需要小时线请用 Interval.MINUTE 拉回再用 BarGenerator 合成。"
)

#: (源 → 周期 → 声明)。新增数据源时在这里补一行，不要在各自 mapping 里散落硬编码。
SOURCE_LABELS: dict[str, dict[Interval, LabelSpec]] = {
    SOURCE_FUTU: {
        Interval.MINUTE: LabelSpec(
            LabelKind.END, Confidence.VERIFIED, _FUTU_INTRADAY_EVIDENCE
        ),
        Interval.HOUR: LabelSpec(
            LabelKind.END, Confidence.VERIFIED, _FUTU_HOUR_EVIDENCE
        ),
        Interval.DAILY: LabelSpec(
            LabelKind.SESSION_DATE, Confidence.VERIFIED, _DAILY_EVIDENCE
        ),
        Interval.WEEKLY: LabelSpec(
            LabelKind.UNKNOWN,
            Confidence.ASSUMED,
            "K_WEEK 的锚点（周一 or 周五）未实测 ⇒ 默认不动。",
        ),
    },
    SOURCE_USMART: {
        Interval.MINUTE: LabelSpec(
            LabelKind.START, Confidence.VERIFIED, _USMART_MINUTE_EVIDENCE
        ),
        Interval.HOUR: LabelSpec(
            LabelKind.UNKNOWN, Confidence.VERIFIED, _USMART_HOUR_EVIDENCE, trusted=False
        ),
        Interval.DAILY: LabelSpec(
            LabelKind.SESSION_DATE, Confidence.VERIFIED, _DAILY_EVIDENCE
        ),
        Interval.WEEKLY: LabelSpec(
            LabelKind.UNKNOWN, Confidence.ASSUMED, "周线锚点未实测 ⇒ 默认不动。"
        ),
    },
    SOURCE_LONGBRIDGE: {
        Interval.MINUTE: LabelSpec(
            LabelKind.START,
            Confidence.VERIFIED,
            "2026-07-24 700.HK 1m 实测(UTC→HKT)：09:30 V=1646424 / 11:59 / 13:00 /"
            " 15:59 / 16:00 CAS，与 uSMART 逐根吻合 ⇒ START。",
        ),
    },
    SOURCE_REPLAY: {
        Interval.MINUTE: LabelSpec(
            LabelKind.START,
            Confidence.STRUCTURAL,
            "回放网关由 vnpy 自己的 BarGenerator 产出 ⇒ 定义上就是 START。",
        ),
    },
}


def label_spec(source: str, interval: Interval) -> LabelSpec:
    """取 (源, 周期) 的标签声明；未登记的一律返回 UNKNOWN（默认不动）。"""
    return SOURCE_LABELS.get(source, {}).get(interval, _UNKNOWN_SPEC)


def is_trusted(source: str, interval: Interval) -> bool:
    """该 (源, 周期) 的数据是否可以落库。"""
    return label_spec(source, interval).trusted


def assert_trusted(source: str, interval: Interval) -> None:
    """不可信就抛，附上实测证据 —— 给 gateway.query_history 当前置闸用。"""
    spec = label_spec(source, interval)
    if not spec.trusted:
        raise ValueError(
            f"{source} 的 {interval.value} 周期数据不可用于落库: {spec.evidence}"
        )


# ---------------------------------------------------------------------------
# 单根归一
# ---------------------------------------------------------------------------
#: 有 session 内栅格意义的周期。日线/周线不在此列（它们是日期标识）。
_PERIODS: dict[Interval, timedelta] = {
    Interval.MINUTE: timedelta(minutes=1),
    Interval.HOUR: timedelta(hours=1),
}


@dataclass(frozen=True, slots=True)
class LabelResult:
    """一次归一的结果 + 可审计的原因。"""

    datetime: datetime
    changed: bool
    note: str
    window: Session | None = None
    drop: bool = False


def _day_windows(
    exchange: Exchange, day: date
) -> tuple[tuple[Session, datetime, datetime], ...]:
    """当天所有 session 窗口（**不合并**，竞价与连续必须分得开）。"""
    tz = market_tz(exchange)
    return tuple(
        (
            session,
            datetime.combine(day, session.start, tzinfo=tz),
            datetime.combine(day, session.end, tzinfo=tz),
        )
        for session in sessions_for(exchange)
    )


def _containing_window(
    moment: datetime, exchange: Exchange
) -> tuple[Session, datetime, datetime] | None:
    """END 标签落在哪个窗口里。

    判据是 ``start < moment <= end``（左开右闭）—— END 标签正好落在窗口的
    右端点上（港股 12:00、美股 16:00），而窗口的左端点属于**上一个**窗口的
    END（港股 09:30 是开盘竞价窗口的 END，不是上午连续的 END）。
    """
    for session, start, end in _day_windows(exchange, moment.date()):
        if start < moment <= end:
            return session, start, end
    return None


def _grid_start(window_start: datetime, moment: datetime, period: timedelta) -> datetime:
    """窗口内小于 ``moment`` 的最大栅格点。

    ``k = ceil((moment - window_start) / period) - 1``，加法在 UTC 上做：一根
    分钟 bar 是 60 个**真实秒**，不是 60 个墙钟秒。美股 session 边界没有一个
    落在 DST 切换小时内，所以这与墙钟加法在实盘上等价，但 UTC 口径在结构上
    不可能算出不存在/二义的本地时刻。
    """
    delta = int((moment - window_start).total_seconds())
    step = int(period.total_seconds())
    if delta <= 0 or step <= 0:
        raise ValueError(
            f"栅格计算前提被破坏: delta={delta}s step={step}s "
            f"(window_start={window_start}, moment={moment})"
        )
    k = -(-delta // step) - 1  # ceil 除法后减一
    tz = window_start.tzinfo
    # timezone.utc, not datetime.UTC: that alias is 3.11+ and this package
    # declares requires-python >=3.10 (sessions.py uses the same spelling).
    start_utc = window_start.astimezone(timezone.utc) + timedelta(seconds=k * step)  # noqa: UP017
    return start_utc.astimezone(tz)


def to_start_label(
    moment: datetime,
    *,
    source: str,
    exchange: Exchange,
    interval: Interval,
    auction_policy: AuctionPolicy = AuctionPolicy.WINDOW_START,
    enabled: bool | None = None,
) -> LabelResult:
    """把一根 bar 的标签归一到 vnpy 的 START 语义。

    非 END 语义、未验证语义、全局开关关闭、无法定位 session 窗口 —— 任一情况
    都返回**原值**并附上 note，绝不猜。
    """
    if moment.tzinfo is None:
        raise ValueError(
            "bar 标签必须是 tz-aware（先过 vnpy_gatewaykit.localize）："
            "裸墙钟时刻无法与市场 session 比较"
        )

    if not _resolve_enabled(enabled):
        return LabelResult(moment, False, f"disabled:{ENV_SWITCH}")

    spec = label_spec(source, interval)
    if not spec.normalizable:
        return LabelResult(
            moment, False, f"no-op:{spec.kind.value}/{spec.confidence.value}"
        )

    period = _PERIODS.get(interval)
    if period is None:
        return LabelResult(moment, False, f"no-op:no-grid-for-{interval.value}")

    return start_label_on_grid(
        moment, exchange=exchange, period=period, auction_policy=auction_policy
    )


def start_label_on_grid(
    moment: datetime,
    *,
    exchange: Exchange,
    period: timedelta,
    auction_policy: AuctionPolicy = AuctionPolicy.WINDOW_START,
) -> LabelResult:
    """把一个 **END(收尾)时刻** 映射到它所属 session 栅格的起始时刻。

    这是 :func:`to_start_label` 的底层原语，区别只在于**周期由调用方显式给**、
    不查 :data:`SOURCE_LABELS` 注册表：

    * :func:`to_start_label` 服务**网关**——(源, 周期) 的语义是我们实测出来的，
      所以先查注册表确认 END + 证据够硬，周期由 ``Interval`` 定死。
    * 本函数服务**语义由调用方声明**的场景（CSV 批量导入：文件是 open 标还是
      close 标只有调用方知道），且周期可以不等于 ``Interval``——一个 5 分钟
      vendor 导出会以 ``Interval.MINUTE`` 落库，栅格步长必须是 5 分钟。

    两边共用同一份栅格数学，所以"裸减一个周期"这个错在任何入口都犯不了：
    港股 60m 标 12:00 的那根覆盖 ``[11:30, 12:00)``（午休截断出的 30 分钟残根），
    裸减 60 分钟得到 11:00 落在上一根的窗口里；本函数按窗口重建栅格得 11:30。
    美股 60m 标 16:00 同理 → 15:30，不是 15:00。

    ``period`` 必须为正。返回的 :class:`LabelResult` 带 note 说明走了哪条分支，
    定位不到窗口时返回原时刻并标 ``unchanged:outside-all-windows``——不猜。
    """
    if moment.tzinfo is None:
        raise ValueError(
            "bar 标签必须是 tz-aware（先过 vnpy_gatewaykit.localize）："
            "裸墙钟时刻无法与市场 session 比较"
        )
    if period <= timedelta(0):
        raise ValueError(f"栅格步长必须为正: period={period}")

    local = moment.astimezone(market_tz(exchange))
    found = _containing_window(local, exchange)
    if found is None:
        # 例如带 extended_time 拉到了 session 表没覆盖的时段。不猜，原样放行。
        return LabelResult(local, local != moment, "unchanged:outside-all-windows")

    session, window_start, _window_end = found

    if session.kind is SessionKind.AUCTION:
        if auction_policy is AuctionPolicy.KEEP:
            return LabelResult(local, local != moment, "auction:keep", session)
        if auction_policy is AuctionPolicy.DROP:
            return LabelResult(local, False, "auction:drop", session, drop=True)
        # WINDOW_START：整段竞价就是一根 bar，它的 START 就是窗口开始。
        return LabelResult(
            window_start, window_start != moment, "auction:window-start", session
        )

    start = _grid_start(window_start, local, period)
    return LabelResult(start, start != moment, "shifted:end->start", session)


# ---------------------------------------------------------------------------
# 整段归一
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class NormalizeReport:
    """一次整段归一的可审计账本。"""

    source: str
    interval: Interval
    total: int = 0
    shifted: int = 0
    unchanged: int = 0
    dropped: int = 0
    auction: int = 0
    outside_window: int = 0
    #: 归一后仍然重名的标签 —— 竞价段拆并方式不同会产生这个，主键去不掉。
    collisions: tuple[tuple[datetime, int], ...] = ()
    notes: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        parts = [
            f"source={self.source}",
            f"interval={self.interval.value}",
            f"total={self.total}",
            f"shifted={self.shifted}",
            f"unchanged={self.unchanged}",
            f"dropped={self.dropped}",
            f"auction={self.auction}",
            f"outside_window={self.outside_window}",
            f"collisions={len(self.collisions)}",
        ]
        return " ".join(parts)


def normalize_bars(
    bars: Sequence[BarData],
    *,
    source: str,
    auction_policy: AuctionPolicy = AuctionPolicy.WINDOW_START,
    enabled: bool | None = None,
) -> tuple[list[BarData], NormalizeReport]:
    """整段归一，返回新列表 + 报告。输入不被修改。

    每根 bar 用自己的 ``interval``；``interval`` 为 None 的行原样放行并计入
    ``unchanged``（vnpy 允许 BarData.interval 为空）。
    """
    intervals = {bar.interval for bar in bars if bar.interval is not None}
    report = NormalizeReport(
        source=source,
        interval=next(iter(intervals)) if len(intervals) == 1 else Interval.TICK,
    )

    out: list[BarData] = []
    for bar in bars:
        report.total += 1
        if bar.interval is None:
            report.unchanged += 1
            report.notes["no-op:interval-is-none"] = (
                report.notes.get("no-op:interval-is-none", 0) + 1
            )
            out.append(bar)
            continue

        result = to_start_label(
            bar.datetime,
            source=source,
            exchange=bar.exchange,
            interval=bar.interval,
            auction_policy=auction_policy,
            enabled=enabled,
        )
        report.notes[result.note] = report.notes.get(result.note, 0) + 1
        if result.note.startswith("auction:"):
            report.auction += 1
        if result.note.endswith("outside-all-windows"):
            report.outside_window += 1

        if result.drop:
            report.dropped += 1
            continue

        if result.changed:
            report.shifted += 1
            new_bar = copy.copy(bar)
            new_bar.datetime = result.datetime
            out.append(new_bar)
        else:
            report.unchanged += 1
            out.append(bar)

    seen: dict[datetime, int] = {}
    for bar in out:
        seen[bar.datetime] = seen.get(bar.datetime, 0) + 1
    report.collisions = tuple(
        sorted((stamp, count) for stamp, count in seen.items() if count > 1)
    )
    return out, report


def relabel_stored_bars(
    bars: Sequence[BarData],
    *,
    source: str,
    from_version: int,
    to_version: int = LABEL_SCHEMA_VERSION,
    auction_policy: AuctionPolicy = AuctionPolicy.WINDOW_START,
) -> tuple[list[BarData], NormalizeReport]:
    """把**已落库**的一段 bar 从旧口径迁到新口径（离线迁移用）。

    只支持 v1(原样透传) → v2(START 归一)；同版本是 no-op；其余组合直接抛，
    绝不"尽力而为"地猜一个方向 —— 迁移方向搞反等于把数据再错一次。
    """
    if from_version == to_version:
        return list(bars), NormalizeReport(
            source=source,
            interval=Interval.TICK,
            total=len(bars),
            unchanged=len(bars),
            notes={"no-op:same-version": len(bars)},
        )
    if (from_version, to_version) != (RAW_LABEL_SCHEMA_VERSION, LABEL_SCHEMA_VERSION):
        raise ValueError(
            f"不支持的标签口径迁移 v{from_version} → v{to_version}；"
            f"只实现了 v{RAW_LABEL_SCHEMA_VERSION} → v{LABEL_SCHEMA_VERSION}"
        )
    return normalize_bars(
        bars, source=source, auction_policy=auction_policy, enabled=True
    )


# ---------------------------------------------------------------------------
# 口径台账：防止新旧口径混在同一段序列里
# ---------------------------------------------------------------------------
class LabelSchemaConflict(RuntimeError):
    """要写入的口径与该序列已落库的口径不一致。"""


@dataclass(frozen=True, slots=True)
class SeriesKey:
    """台账的主键：一段"同源同标的同周期"的连续序列。"""

    source: str
    symbol: str
    exchange: Exchange
    interval: Interval

    @property
    def text(self) -> str:
        return f"{self.source}|{self.symbol}.{self.exchange.value}|{self.interval.value}"

    @classmethod
    def parse(cls, text: str) -> SeriesKey:
        try:
            source, vt_symbol, interval = text.split("|")
            symbol, _, exchange = vt_symbol.rpartition(".")
            return cls(source, symbol, Exchange(exchange), Interval(interval))
        except (ValueError, KeyError) as exc:
            raise ValueError(f"非法的 SeriesKey 文本: {text!r}") from exc

    @classmethod
    def of(cls, bar: BarData, source: str) -> SeriesKey:
        if bar.interval is None:
            raise ValueError(f"bar {bar.vt_symbol} 没有 interval，无法定位台账主键")
        return cls(source, bar.symbol, bar.exchange, bar.interval)


class LabelLedger:
    """记录每段序列是用哪个标签口径落的库。

    这是"历史已落库数据与新数据不会混"的执行体。归一会改 datetime，所以新旧
    口径混在同一段序列里既不是重复行也不是一致行 —— 主键去重发现不了，图表和
    回测会静默读到两套错位的 bar。台账在写入前把这件事变成一个异常。

    存成独立 JSON（不动 vnpy 的 DB schema），原子替换写入。
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = Path(path)
        self._data: dict[str, int] = self._load()

    @property
    def path(self) -> Path:
        return self._path

    def _load(self) -> dict[str, int]:
        if not self._path.exists():
            return {}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"标签口径台账 {self._path} 已损坏: {exc}。"
                f"绝不当作空台账继续 —— 那会让守卫静默失效。请人工修复或删除后重新 bootstrap。"
            ) from exc
        if not isinstance(raw, dict):
            raise ValueError(f"标签口径台账 {self._path} 顶层应是对象，实得 {type(raw).__name__}")
        out: dict[str, int] = {}
        for key, value in raw.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"标签口径台账 {self._path} 的 {key!r} 版本号非整数: {value!r}")
            out[str(key)] = value
        return out

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self._data, ensure_ascii=False, indent=2, sort_keys=True)
        fd, tmp = tempfile.mkstemp(dir=str(self._path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.write("\n")
            os.replace(tmp, self._path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def version_of(self, key: SeriesKey) -> int | None:
        return self._data.get(key.text)

    def keys(self) -> tuple[SeriesKey, ...]:
        return tuple(SeriesKey.parse(text) for text in sorted(self._data))

    def declare(self, key: SeriesKey, version: int, *, overwrite: bool = False) -> None:
        """登记某段序列的口径。已登记且不同版本时必须显式 overwrite。"""
        current = self._data.get(key.text)
        if current is not None and current != version and not overwrite:
            raise LabelSchemaConflict(
                f"{key.text} 已登记为 v{current}，不能改登记为 v{version}；"
                f"迁移完成后请用 overwrite=True 重新登记"
            )
        self._data[key.text] = version
        self._save()

    def bootstrap(
        self, keys: Iterable[SeriesKey], version: int, *, overwrite: bool = False
    ) -> None:
        """一次性声明"这些序列现存数据是哪个口径"。上线归一前必须先跑这个。"""
        for key in keys:
            current = self._data.get(key.text)
            if current is not None and current != version and not overwrite:
                raise LabelSchemaConflict(
                    f"{key.text} 已登记为 v{current}，bootstrap 到 v{version} 需要 overwrite=True"
                )
            self._data[key.text] = version
        self._save()

    def assert_compatible(
        self, key: SeriesKey, version: int, *, require_declared: bool = False
    ) -> None:
        """写入前的闸。口径不一致就抛，绝不合并。"""
        current = self._data.get(key.text)
        if current is None:
            if require_declared:
                raise LabelSchemaConflict(
                    f"{key.text} 未在标签口径台账中登记；已有历史数据的序列必须先 "
                    f"bootstrap(RAW_LABEL_SCHEMA_VERSION) 或 declare(LABEL_SCHEMA_VERSION)，"
                    f"否则新旧口径会混在同一段序列里"
                )
            return
        if current != version:
            raise LabelSchemaConflict(
                f"{key.text} 已落库口径为 v{current}，本次要写 v{version}。"
                f"标签口径不同 = 同一根 bar 落在不同时间戳上，主键去重发现不了。"
                f"请先用 relabel_stored_bars() 迁移整段，再 declare(..., overwrite=True)"
            )
