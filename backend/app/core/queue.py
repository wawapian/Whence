"""后台任务队列：命名车道（record 串行保序 / doc 并行）+ 内存任务记录（future/result/error）。

- `record` 车道 = 1 工人 FIFO：记录整理（打码/抽取/事实/索引/材料）必须保序——
  双时态「后说的覆盖先说的」依赖整理顺序 = 用户提交顺序，多线程乱序会翻错状态；
- `doc` 车道 = 4 工人并行：文档完整解析（解析/切片/嵌入/表格/caption），互相独立可并行。
"""
import uuid
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor


_executors = {
    "record": ThreadPoolExecutor(max_workers=1),
    "doc": ThreadPoolExecutor(max_workers=4),
}
_tasks: dict[str, dict] = OrderedDict()
_lock = threading.Lock()


def submit(fn, *args, lane: str = "doc", **kwargs) -> str:
    """提交任务到指定车道，返回 task_id（uuid4 hex）；未知车道按 doc 处理。

    包装函数把返回值或异常捕获进任务记录，供 status 查询。
    """
    task_id = uuid.uuid4().hex
    record: dict = {"future": None, "result": None, "error": None}
    def _run() -> None:
        try:
            record["result"] = fn(*args, **kwargs)
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"

    record["future"] = _executors.get(lane, _executors["doc"]).submit(_run)
    with _lock:
        _tasks[task_id] = record
        # 限制已结束任务记录数量，释放 Future、结果和异常引用。
        for old_id in list(_tasks):
            if len(_tasks) <= 1024:
                break
            if _tasks[old_id]["future"].done():
                _tasks.pop(old_id)
    return task_id


def status(task_id: str) -> str:
    """查询任务状态：processing（未完成）| failed（有异常）| done（完成）。"""
    record = _tasks.get(task_id)
    if record is None or not record["future"].done():
        return "processing"
    if record["error"] is not None:
        return "failed"
    return "done"
