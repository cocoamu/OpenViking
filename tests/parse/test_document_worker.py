"""Real subprocess lifecycle/budget tests; PDF/Office quality is tested separately."""

import asyncio
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


SOURCE = Path(__file__).parents[2] / "openviking/parse/document_worker.py"
spec = importlib.util.spec_from_file_location("document_worker_under_test", SOURCE)
worker = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = worker
spec.loader.exec_module(worker)

FAKE_RUNNER = '''
import json,os,signal,time
from pathlib import Path
import sys
req,out=map(Path,sys.argv[1:3]);r=json.loads(req.read_text());mode=Path(r['source']).stem
Path(r['media_dir']).mkdir(parents=True,exist_ok=True)
if mode=='ignore':signal.signal(signal.SIGTERM,signal.SIG_IGN)
Path(r['media_dir'],'started').write_text(str(os.getpid()))
if mode in ['sleep','ignore']:
 time.sleep(30)
if mode=='resident':
 x=bytearray(80*1024*1024)
 time.sleep(30)
if mode=='exit':os._exit(23)
if mode=='burst':
 import resource
 resource.setrlimit(resource.RLIMIT_AS,(r['memory_bytes'],r['memory_bytes']))
 try:
  allocated=bytearray(128*1024*1024)
  blocked=False
 except MemoryError:blocked=True
 out.write_text(json.dumps({'status':'ok','result':{'markdown':'bounded document','meta':{'allocationBlockedBeforeRSSGrowth':blocked}},'stats':{'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}}));sys.exit(0)
if mode=='bad':out.write_text('{"status":"ok","result":{"markdown":23}}');sys.exit(0)
if mode=='memory':
 import resource
 resource.setrlimit(resource.RLIMIT_AS,(96*1024*1024,96*1024*1024))
 try:x=bytearray(256*1024*1024)
 except MemoryError:
  out.write_text(json.dumps({'status':'error','error':{'type':'MemoryError','message':'bounded worker allocation'}}));sys.exit(1)
out.write_text(json.dumps({'status':'ok','result':{'markdown':'same document','meta':{'pages_processed':88}},'stats':{'peak_rss_kib':10000}}))
'''


class MemoryAdmissionTests(unittest.TestCase):
    def setUp(self):
        worker._ACTIVE_WORKERS = 0
        self.settings = worker.WorkerSettings(memory_bytes=30, max_workers=2)

    def tearDown(self):
        worker._ACTIVE_WORKERS = 0

    def test_pressure_includes_future_reserved_memory(self):
        with patch.object(worker, "_memory_snapshot", return_value=(30, 100)):
            self.assertTrue(worker._try_acquire(self.settings))
            # 30 resident + two 30-byte reservations > 85-byte budget.
            self.assertFalse(worker._try_acquire(self.settings))

    def test_worker_cap_is_process_global_across_threads(self):
        settings = worker.WorkerSettings(memory_bytes=1, max_workers=1)
        results = []
        with patch.object(worker, "_memory_snapshot", return_value=None):
            threads = [threading.Thread(target=lambda: results.append(worker._try_acquire(settings)))
                       for _ in range(12)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        self.assertEqual(results.count(True), 1)
        self.assertEqual(worker._ACTIVE_WORKERS, 1)

    def test_four_gib_container_reserves_physical_not_virtual_memory(self):
        settings = worker.WorkerSettings()
        with patch.object(worker, "_memory_snapshot", return_value=(1400*worker._MIB, 4096*worker._MIB)):
            self.assertTrue(worker._try_acquire(settings))
        worker._ACTIVE_WORKERS = 0
        with patch.object(worker, "_memory_snapshot", return_value=(2600*worker._MIB, 4096*worker._MIB)):
            self.assertFalse(worker._try_acquire(settings))

    def test_cgroup_v2_and_v1_formats(self):
        for values, expected in [
            ({"/sys/fs/cgroup/memory.current":"45", "/sys/fs/cgroup/memory.max":"100"},(45,100)),
            ({"/sys/fs/cgroup/memory/memory.usage_in_bytes":"50",
              "/sys/fs/cgroup/memory/memory.limit_in_bytes":"200"},(50,200)),
            ({"/sys/fs/cgroup/memory.current":"45", "/sys/fs/cgroup/memory.max":"max"},None),
        ]:
            def read(path, *args, **kwargs):
                if str(path) not in values:
                    raise FileNotFoundError
                return values[str(path)]
            with patch.object(Path, "read_text", read):
                self.assertEqual(worker._memory_snapshot(), expected)

    def test_isolation_threshold_and_platform_escape(self):
        with patch.object(worker.sys, "platform", "linux"), patch.object(Path,"stat", return_value=SimpleNamespace(st_size=4*1024*1024)):
            self.assertTrue(worker.should_isolate(Path("original.pdf")))
            with patch.dict(os.environ,{"OPENVIKING_DOCUMENT_ISOLATION":"0"}):
                self.assertFalse(worker.should_isolate(Path("original.pdf")))
        with patch.object(worker.sys,"platform","darwin"):
            self.assertFalse(worker.should_isolate(Path("original.pdf")))

    def test_small_compressed_documents_are_isolated_before_decoding(self):
        suffixes = (".pdf", ".PDF", ".doc", ".docx", ".docm", ".odt", ".rtf",
                    ".ppt", ".pptx", ".pptm", ".pps", ".ppsx", ".ppsm", ".pot", ".odp",
                    ".xls", ".xlsx", ".xlsm", ".xlsb", ".ods", ".epub")
        with patch.object(worker.sys, "platform", "linux"), \
             patch.object(Path, "stat", return_value=SimpleNamespace(st_size=1024)):
            for suffix in suffixes:
                with self.subTest(suffix=suffix):
                    self.assertTrue(worker.should_isolate(Path("compressed" + suffix)))
            self.assertFalse(worker.should_isolate(Path("plain.csv")))
            self.assertFalse(worker.should_isolate(Path("plain.txt")))
        with patch.object(worker.sys, "platform", "linux"), \
             patch.object(Path, "stat", side_effect=FileNotFoundError):
            self.assertFalse(worker.should_isolate(Path("missing.pdf")))

    def test_invalid_budgets_fail_explicitly(self):
        with patch.dict(os.environ,{"OPENVIKING_DOCUMENT_WORKER_MEMORY_MB":"0"}):
            with self.assertRaises(ValueError):
                worker.WorkerSettings.from_environment()
        with patch.dict(os.environ,{"OPENVIKING_DOCUMENT_WORKER_RESIDENT_MB":"0"}):
            with self.assertRaises(ValueError):
                worker.WorkerSettings.from_environment()
        for value in ("nan", "inf"):
            with patch.dict(os.environ, {"OPENVIKING_DOCUMENT_WORKER_TIMEOUT": value}):
                with self.assertRaises(ValueError):
                    worker.WorkerSettings.from_environment()


@unittest.skipUnless(os.name == "posix", "Process-group lifecycle requires POSIX")
class ParentReclaimTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        worker._ACTIVE_WORKERS = 0
        worker._NEXT_RECLAIM_AT = 0

    async def asyncTearDown(self):
        worker._ACTIVE_WORKERS = 0
        worker._NEXT_RECLAIM_AT = 0

    async def test_pressure_reclaim_allows_slot_only_after_memory_really_drops(self):
        settings = worker.WorkerSettings(memory_bytes=30, resident_bytes=30)
        used = [90]
        def reclaim():
            used[0] = 40
        with patch.object(worker.sys, "platform", "linux"), \
             patch.object(worker, "_memory_snapshot", side_effect=lambda: (used[0], 100)), \
             patch.object(worker, "_reclaim_parent_memory", side_effect=reclaim) as action:
            async with worker._worker_slot(settings):
                self.assertEqual(worker._ACTIVE_WORKERS, 1)
                self.assertEqual(action.call_count, 1)
        self.assertEqual(worker._ACTIVE_WORKERS, 0)

    async def test_reclaim_failure_to_free_memory_never_bypasses_admission(self):
        settings = worker.WorkerSettings(admission_timeout_seconds=0.1)
        with patch.object(worker.sys, "platform", "linux"), \
             patch.object(worker, "_memory_snapshot", return_value=(99, 100)), \
             patch.object(worker, "_reclaim_parent_memory") as action:
            with self.assertRaisesRegex(worker.DocumentWorkerError, "ADMISSION_TIMEOUT"):
                async with worker._worker_slot(settings):
                    self.fail("Unsafe slot was admitted")
            self.assertEqual(action.call_count, 1)
        self.assertEqual(worker._ACTIVE_WORKERS, 0)

    async def test_low_pressure_and_running_workers_do_not_trigger_gc(self):
        settings = worker.WorkerSettings(memory_bytes=30, resident_bytes=30)
        with patch.object(worker.sys, "platform", "linux"), \
             patch.object(worker, "_reclaim_parent_memory") as action:
            with patch.object(worker, "_memory_snapshot", return_value=(10, 100)):
                await worker._maybe_reclaim_parent_memory(settings)
            worker._ACTIVE_WORKERS = 1
            with patch.object(worker, "_memory_snapshot", return_value=(99, 100)):
                await worker._maybe_reclaim_parent_memory(settings)
            action.assert_not_called()

    async def test_concurrent_pressure_probes_share_reclaim_cooldown(self):
        with patch.object(worker.sys, "platform", "linux"), \
             patch.object(worker, "_memory_snapshot", return_value=(99, 100)), \
             patch.object(worker, "_reclaim_parent_memory") as action:
            await asyncio.gather(*[worker._maybe_reclaim_parent_memory(worker.WorkerSettings())
                                   for _ in range(8)])
            self.assertEqual(action.call_count, 1)

    async def test_non_linux_reclaim_is_skipped(self):
        with patch.object(worker.sys, "platform", "darwin"), \
             patch.object(worker, "_reclaim_parent_memory") as action:
            await worker._maybe_reclaim_parent_memory(worker.WorkerSettings())
            action.assert_not_called()

    async def test_allocator_reclaim_preserves_live_data(self):
        live = bytearray(b"keep this document")
        trim = MagicMock(return_value=1)
        with patch.object(worker.ctypes, "CDLL", return_value=SimpleNamespace(malloc_trim=trim)), \
             patch.object(worker.gc, "collect") as collect:
            worker._reclaim_parent_memory()
        collect.assert_called_once()
        trim.assert_called_once_with(0)
        self.assertEqual(live, b"keep this document")

    async def test_missing_allocator_trim_remains_safe(self):
        with patch.object(worker.ctypes, "CDLL", side_effect=OSError("unavailable")), \
             patch.object(worker.gc, "collect") as collect:
            worker._reclaim_parent_memory()
        collect.assert_called_once()


@unittest.skipUnless(os.name == "posix", "Process-group lifecycle requires POSIX")
class WorkerLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "_document_worker_runner.py").write_text(FAKE_RUNNER)
        worker._ACTIVE_WORKERS = 0
        self.patches = [
            patch.object(worker,"__file__",str(self.root/"document_worker.py")),
            patch.object(worker,"_memory_snapshot",return_value=None),
            patch.object(worker.WorkerSettings,"from_environment",return_value=worker.WorkerSettings(timeout_seconds=5)),
        ]
        for item in self.patches:
            item.start()

    async def asyncTearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.assertEqual(worker._ACTIVE_WORKERS, 0)
        self.temp.cleanup()

    def run_worker(self, mode="success"):
        return worker.convert_document_in_worker(
            "pdf",self.root/(mode+".pdf"), storage=SimpleNamespace(media_dir=self.root/"media"),
            resource_name="original",
        )

    async def started(self):
        for _ in range(200):
            path=self.root/"media/started"
            if path.exists():
                return int(path.read_text())
            await asyncio.sleep(0.01)
        self.fail("worker did not start")

    def assert_reaped(self,pid):
        with self.assertRaises(ProcessLookupError):
            os.kill(pid,0)

    async def test_success_preserves_payload_and_reaps_child(self):
        result=await self.run_worker()
        self.assertEqual(result,{"markdown":"same document","meta":{"pages_processed":88}})
        self.assert_reaped(await self.started())

    async def test_abnormal_worker_exit_is_contained(self):
        with self.assertRaisesRegex(worker.DocumentWorkerError,"code 23"):
            await self.run_worker("exit")
        self.assertEqual((await self.run_worker())["markdown"],"same document")

    async def test_bad_protocol_cannot_mark_conversion_successful(self):
        with self.assertRaisesRegex(worker.DocumentWorkerError,"PROTOCOL"):
            await self.run_worker("bad")

    @unittest.skipUnless(sys.platform=="linux","Linux address-space allocation limit")
    async def test_real_memory_limit_fails_only_worker_and_next_file_runs(self):
        with self.assertRaisesRegex(worker.DocumentWorkerError,"MemoryError"):
            await self.run_worker("memory")
        self.assertEqual((await self.run_worker())["markdown"],"same document")

    @unittest.skipUnless(sys.platform=="linux", "Linux hard address-space allocation guard")
    async def test_burst_allocation_is_rejected_before_resident_budget_is_exceeded(self):
        settings = worker.WorkerSettings(memory_bytes=256*worker._MIB,
                                         resident_bytes=64*worker._MIB)
        with patch.object(worker.WorkerSettings, "from_environment", return_value=settings):
            result = await self.run_worker("burst")
        self.assertTrue(result["meta"]["allocationBlockedBeforeRSSGrowth"])
        self.assert_reaped(await self.started())
        self.assertEqual((await self.run_worker())["markdown"], "same document")

    async def test_timeout_kills_and_reaps_child(self):
        settings=worker.WorkerSettings(timeout_seconds=0.15)
        with patch.object(worker.WorkerSettings,"from_environment",return_value=settings):
            with self.assertRaisesRegex(worker.DocumentWorkerError,"WORKER_TIMEOUT"):
                await self.run_worker("sleep")
        self.assert_reaped(await self.started())

    async def test_container_pressure_stops_child_and_next_file_can_run(self):
        task = asyncio.create_task(self.run_worker("sleep"))
        pid = await self.started()
        with patch.object(worker, "_memory_snapshot", return_value=(90,100)):
            with self.assertRaisesRegex(worker.DocumentWorkerError,"WORKER_MEMORY_LIMIT"):
                await task
        self.assert_reaped(pid)
        self.assertEqual((await self.run_worker())["markdown"],"same document")

    @unittest.skipUnless(sys.platform=="linux","Linux resident memory accounting")
    async def test_real_resident_limit_stops_and_reaps_only_worker(self):
        settings = worker.WorkerSettings(resident_bytes=32*worker._MIB)
        with patch.object(worker.WorkerSettings,"from_environment",return_value=settings):
            with self.assertRaisesRegex(worker.DocumentWorkerError,"WORKER_MEMORY_LIMIT"):
                await self.run_worker("resident")
        self.assert_reaped(await self.started())
        self.assertEqual((await self.run_worker())["markdown"],"same document")

    async def test_cancel_reaps_child_before_releasing_admission(self):
        task=asyncio.create_task(self.run_worker("sleep"));pid=await self.started()
        self.assertEqual(worker._ACTIVE_WORKERS,1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assert_reaped(pid)
        self.assertEqual(worker._ACTIVE_WORKERS,0)

    async def test_late_spawn_cancellation_still_reaps_child(self):
        create=asyncio.create_subprocess_exec
        async def delayed(*args,**kwargs):
            proc=await create(*args,**kwargs)
            await asyncio.sleep(0.1)
            return proc
        with patch.object(worker.asyncio,"create_subprocess_exec",side_effect=delayed):
            task=asyncio.create_task(self.run_worker("sleep"));pid=await self.started()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assert_reaped(pid)

    async def test_sigterm_resistant_worker_is_force_killed(self):
        task=asyncio.create_task(self.run_worker("ignore"));pid=await self.started()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assert_reaped(pid)

    async def test_cancel_waiting_task_does_not_release_another_task_slot(self):
        task=asyncio.create_task(self.run_worker("sleep"));pid=await self.started()
        waiting=asyncio.create_task(self.run_worker());await asyncio.sleep(0.05)
        waiting.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiting
        self.assertEqual(worker._ACTIVE_WORKERS,1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assert_reaped(pid)

    async def test_pressure_admission_times_out_without_starting_child(self):
        with patch.object(worker,"_memory_snapshot",return_value=(99,100)), \
             patch.object(worker.WorkerSettings,"from_environment",return_value=worker.WorkerSettings(admission_timeout_seconds=0.1)):
            with self.assertRaisesRegex(worker.DocumentWorkerError,"ADMISSION_TIMEOUT"):
                await self.run_worker()
        self.assertFalse((self.root/"media/started").exists())


@unittest.skipUnless(sys.platform == "linux", "Linux parent-death signal")
class ParentDeathTests(unittest.TestCase):
    def test_real_protection_kills_worker_when_parent_is_killed(self):
        runner = SOURCE.with_name("_document_worker_runner.py")
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "child.json"
            supervisor = '''
import importlib.util,json,os,sys,time
from pathlib import Path
spec=importlib.util.spec_from_file_location('runner',sys.argv[1])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
parent=os.getpid();pid=os.fork()
if pid==0:
 module._protect_worker(128*1024*1024,parent)
 Path(sys.argv[2]).write_text(json.dumps({'pid':os.getpid()}))
 time.sleep(30)
else:time.sleep(30)
'''
            process = subprocess.Popen([sys.executable, "-c", supervisor, str(runner), str(marker)],
                                       start_new_session=True, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL)
            try:
                deadline = time.monotonic() + 5
                while not marker.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(marker.exists(), "protected worker did not start")
                pid = json.loads(marker.read_text())["pid"]
                process.kill(); process.wait(timeout=5)
                status = Path(f"/proc/{pid}/status")
                deadline = time.monotonic() + 5
                while status.exists() and time.monotonic() < deadline:
                    if "State:\tZ" in status.read_text():
                        break
                    time.sleep(0.01)
                self.assertTrue(not status.exists() or "State:\tZ" in status.read_text(),
                                "worker survived its parent")
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)


if __name__=="__main__":
    unittest.main()
