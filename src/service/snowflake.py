import threading
import time
from dataclasses import dataclass

# 自定义起始时间（毫秒）
EPOCH = 1700000000000  # 2023-11-14

# 位数分配
WORKER_ID_BITS = 5
DATACENTER_ID_BITS = 5
SEQUENCE_BITS = 12

# 上限
MAX_WORKER_ID = -1 ^ (-1 << WORKER_ID_BITS)            # 31
MAX_DATACENTER_ID = -1 ^ (-1 << DATACENTER_ID_BITS)    # 31
MAX_SEQUENCE = -1 ^ (-1 << SEQUENCE_BITS)              # 4095

# 位移
WORKER_ID_SHIFT = SEQUENCE_BITS                                 # 12
DATACENTER_ID_SHIFT = SEQUENCE_BITS + WORKER_ID_BITS             # 17
TIMESTAMP_SHIFT = SEQUENCE_BITS + WORKER_ID_BITS + DATACENTER_ID_BITS  # 22


@dataclass
class Snowflake:
    datacenter_id: int
    worker_id: int
    sequence: int = 0
    last_timestamp: int = -1
    _lock: threading.Lock = None

    def __post_init__(self):
        if self.datacenter_id < 0 or self.datacenter_id > MAX_DATACENTER_ID:
            raise ValueError(f"datacenter_id out of range: {self.datacenter_id}")
        if self.worker_id < 0 or self.worker_id > MAX_WORKER_ID:
            raise ValueError(f"worker_id out of range: {self.worker_id}")
        self._lock = threading.Lock()

    def _current_ms(self) -> int:
        return int(time.time() * 1000)

    def _wait_next_ms(self, last_ts: int) -> int:
        """等到下一毫秒"""
        ts = self._current_ms()
        while ts <= last_ts:
            ts = self._current_ms()
        return ts

    def next_id(self) -> int:
        with self._lock:
            ts = self._current_ms()

            # 1. 时钟回拨保护
            if ts < self.last_timestamp:
                raise RuntimeError(
                    f"Clock moved backwards! {self.last_timestamp - ts} ms"
                )

            # 2. 同毫秒内序列号递增
            if ts == self.last_timestamp:
                self.sequence = (self.sequence + 1) & MAX_SEQUENCE
                if self.sequence == 0:
                    # 序列号用完，等下一毫秒
                    ts = self._wait_next_ms(self.last_timestamp)
            else:
                self.sequence = 0

            self.last_timestamp = ts

            # 3. 拼装 ID
            return (
                ((ts - EPOCH) << TIMESTAMP_SHIFT) |
                (self.datacenter_id << DATACENTER_ID_SHIFT) |
                (self.worker_id << WORKER_ID_SHIFT) |
                self.sequence
            )


# ============== 单例工厂 ==============

_snowflake_instance: Snowflake = None


def get_snowflake(
    datacenter_id: int = 1,
    worker_id: int = 1,
    force_new: bool = False,
) -> Snowflake:
    global _snowflake_instance
    if force_new or _snowflake_instance is None:
        _snowflake_instance = Snowflake(datacenter_id, worker_id)
    return _snowflake_instance