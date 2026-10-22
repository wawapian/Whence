"""后台队列命名车道：record 串行保序（双时态不乱序的前提）/ 任务状态查询。"""
import time

from app.core import queue


def test_record_lane_runs_serially_in_order():
    """单工人 FIFO：完成顺序 = 提交顺序（记录整理保序的根保障）。"""
    done = []

    def task(i):
        time.sleep(0.01)
        done.append(i)

    for i in range(5):
        queue.submit(task, i, lane="record")

    deadline = time.time() + 5
    while len(done) < 5 and time.time() < deadline:
        time.sleep(0.02)
    assert done == [0, 1, 2, 3, 4]


def test_status_reports_done_failed_and_unknown():
    ok = queue.submit(lambda: 42, lane="record")

    def _boom():
        raise RuntimeError("boom")

    bad = queue.submit(_boom, lane="record")
    deadline = time.time() + 5
    while time.time() < deadline and (queue.status(ok) != "done" or queue.status(bad) != "failed"):
        time.sleep(0.02)
    assert queue.status(ok) == "done"
    assert queue.status(bad) == "failed"
    assert queue.status("no-such-task") == "processing"
