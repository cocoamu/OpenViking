"""Reproduce native parsing starvation from reclaimable cgroup file cache."""

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SOURCE = Path(__file__).parents[2] / "openviking/parse/document_worker.py"
spec = importlib.util.spec_from_file_location("document_worker_cache_test", SOURCE)
worker = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = worker
spec.loader.exec_module(worker)
GIB = 1024 ** 3


class CacheAdmissionTests(unittest.TestCase):
    def setUp(self):
        worker._ACTIVE_WORKERS = 0

    def tearDown(self):
        worker._ACTIVE_WORKERS = 0

    def readings(self, used, stats, version=2):
        if version == 2:
            root = "/sys/fs/cgroup/"
            values = {root + "memory.current": str(used), root + "memory.max": str(8 * GIB)}
        else:
            root = "/sys/fs/cgroup/memory/"
            values = {root + "memory.usage_in_bytes": str(used),
                      root + "memory.limit_in_bytes": str(8 * GIB)}
        if stats is not None:
            values[root + "memory.stat"] = stats

        def read(path, *args, **kwargs):
            if str(path) not in values:
                raise FileNotFoundError(str(path))
            return values[str(path)]
        return patch.object(Path, "read_text", read)

    def stats(self, inactive=4 * GIB, dirty=0, writeback=0, shmem=0, version=2):
        fields = {"inactive_file": inactive, "file": 5 * GIB,
                  "file_dirty": dirty, "file_writeback": writeback, "shmem": shmem}
        if version == 1:
            fields = {"total_" + {"file": "cache", "file_dirty": "dirty",
                                "file_writeback": "writeback"}.get(k, k): v
                      for k, v in fields.items()}
        return "\n".join(f"{k} {v}" for k, v in fields.items())

    def test_cache_pressure_admits_worker_and_preserves_single_worker_limit(self):
        # 6.5 GiB charged, including 4 GiB inactive clean file cache, in an 8 GiB Pod.
        # Old admission starves for 900 seconds despite just 2.5 GiB working set.
        for version in (1, 2):
            with self.subTest(version=version), self.readings(13 * GIB // 2,
                    self.stats(version=version), version):
                worker._ACTIVE_WORKERS = 0
                self.assertTrue(worker._try_acquire(worker.WorkerSettings()))
                self.assertFalse(worker._try_acquire(worker.WorkerSettings()))

    def test_only_clean_inactive_file_cache_is_discounted(self):
        with self.readings(7 * GIB, self.stats(dirty=GIB, writeback=GIB // 2, shmem=GIB // 2)):
            self.assertEqual(worker._memory_snapshot(), (5 * GIB, 8 * GIB))

    def test_dirty_shmem_and_active_cache_cannot_bypass_pressure_guard(self):
        for stats in (self.stats(inactive=0), self.stats(dirty=4 * GIB),
                      self.stats(shmem=4 * GIB), self.stats(writeback=4 * GIB)):
            with self.subTest(stats=stats), self.readings(7 * GIB, stats):
                self.assertFalse(worker._try_acquire(worker.WorkerSettings()))

    def test_missing_invalid_and_incomplete_stats_keep_raw_usage(self):
        for stats in (None, "inactive_file bad", "inactive_file -1", "inactive_file 1000000"):
            with self.subTest(stats=stats), self.readings(7 * GIB, stats):
                self.assertEqual(worker._memory_snapshot(), (7 * GIB, 8 * GIB))
                self.assertFalse(worker._try_acquire(worker.WorkerSettings()))


if __name__ == "__main__":
    unittest.main()
