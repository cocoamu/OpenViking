"""Consumed handoffs recover through normal exclusive acquisition only."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


class LockAcquisitionError(Exception):
    pass


class SemanticHandoffRecoveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.agfs = types.SimpleNamespace(
            pathlock_adopt=AsyncMock(), pathlock_acquire_batch=AsyncMock(),
            pathlock_release=AsyncMock(), pathlock_as_borrowed=AsyncMock())
        modules = {
            'openviking.storage.errors': types.SimpleNamespace(LockAcquisitionError=LockAcquisitionError),
            'openviking.storage.internal_names': types.SimpleNamespace(MULTIWRITE_PATH_LOCK_FILE='.tree.lock'),
            'openviking.storage.viking_fs': types.SimpleNamespace(get_viking_fs=lambda: types.SimpleNamespace(_async_agfs=self.agfs)),
            'openviking_cli.utils.logger': types.SimpleNamespace(get_logger=lambda name: types.SimpleNamespace(info=lambda *args: None)),
        }
        source = Path(__file__).parents[2] / 'openviking/storage/queuefs/semantic_lock.py'
        spec = importlib.util.spec_from_file_location('semantic_lock_under_test', source)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {**modules, spec.name: module}):
            spec.loader.exec_module(module)
        self.scope = module.SemanticLockScope
        self.coverage = [{'path': '/tenant/resource', 'kind': 'tree'}, {'path': '/tenant/exact', 'kind': 'path'}]
        self.handoff = {'owner_id': 'old-owner', 'covered_paths': self.coverage}
        self.fresh = {'owner_id': 'fresh-owner'}
        self.agfs.pathlock_acquire_batch.return_value = self.fresh

    def consumed(self):
        self.agfs.pathlock_adopt.side_effect = LockAcquisitionError("handoff/adopt failed: pathlock handoff for owner 'old-owner' was already adopted")

    async def test_consumed_handoff_reacquires_exact_coverage_and_releases_fresh_lease(self):
        self.consumed()
        scope = await self.scope.resolve(self.handoff)
        self.agfs.pathlock_acquire_batch.assert_awaited_once_with(self.coverage)
        self.agfs.pathlock_release.assert_not_awaited()
        await scope.close()
        self.agfs.pathlock_release.assert_awaited_once_with(self.fresh)

    async def test_existing_owner_keeps_exclusion_when_handoff_consumed(self):
        self.consumed()
        self.agfs.pathlock_acquire_batch.side_effect = LockAcquisitionError('covered path is busy')
        with self.assertRaisesRegex(LockAcquisitionError, 'busy'):
            await self.scope.resolve(self.handoff)
        self.agfs.pathlock_acquire_batch.assert_awaited_once_with(self.coverage)
        self.agfs.pathlock_release.assert_not_awaited()

    async def test_unrelated_adoption_errors_do_not_acquire_or_release(self):
        for message in ('permission denied', 'invalid handoff token', 'other operation was already adopted'):
            with self.subTest(message=message):
                self.agfs.pathlock_adopt.side_effect = LockAcquisitionError(message)
                with self.assertRaises(LockAcquisitionError):
                    await self.scope.resolve(self.handoff)
        self.agfs.pathlock_acquire_batch.assert_not_awaited()
        self.agfs.pathlock_release.assert_not_awaited()

    async def test_live_handoff_is_adopted_without_reacquisition(self):
        self.agfs.pathlock_adopt.return_value = self.fresh
        scope = await self.scope.resolve(self.handoff)
        self.agfs.pathlock_acquire_batch.assert_not_awaited()
        await scope.close()
        self.agfs.pathlock_release.assert_awaited_once_with(self.fresh)

    async def test_borrowed_caller_lease_is_never_released(self):
        self.agfs.pathlock_as_borrowed.return_value = self.fresh
        scope = await self.scope.resolve(None, caller_lock=self.fresh)
        await scope.close()
        self.agfs.pathlock_as_borrowed.assert_awaited_once_with(self.fresh)
        self.agfs.pathlock_adopt.assert_not_awaited()
        self.agfs.pathlock_acquire_batch.assert_not_awaited()
        self.agfs.pathlock_release.assert_not_awaited()

    async def test_existing_stale_owner_recovery_is_preserved(self):
        self.agfs.pathlock_adopt.side_effect = LockAcquisitionError('lease is no longer owned by old-owner')
        scope = await self.scope.resolve(self.handoff)
        self.assertEqual(scope.lock, self.fresh)
        self.agfs.pathlock_acquire_batch.assert_awaited_once_with(self.coverage)

    async def test_consumed_legacy_handoff_requires_real_coverage(self):
        self.consumed()
        with self.assertRaises(LockAcquisitionError):
            await self.scope.resolve({'owner_id': 'old-owner', 'lock_paths': []})
        self.agfs.pathlock_acquire_batch.assert_not_awaited()

    async def test_consumed_legacy_tree_coverage_is_recovered(self):
        self.consumed()
        scope = await self.scope.resolve({'owner_id': 'old-owner', 'lock_paths': ['/tenant/resource/.tree.lock']})
        self.agfs.pathlock_acquire_batch.assert_awaited_once_with([{'path': '/tenant/resource', 'kind': 'tree'}])
        await scope.close()

    async def test_message_and_caller_lease_are_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            await self.scope.resolve(self.handoff, caller_lock=self.fresh)
        self.agfs.pathlock_adopt.assert_not_awaited()

