"""Tests for T-factory subfactory stage-1 redistribution."""

from src.ds import TFactoryPool
from src.t_cultivation.util import redistribute_stage1_successes


def test_redistribute_one_spare_to_empty_factory():
    pool = TFactoryPool(2, num_subfactories=2)
    pool.set_locations([(0, 0), (10, 0)])
    f0 = pool.get_factory_by_id(0)
    f1 = pool.get_factory_by_id(1)
    f0.subfactories[0].stage_1_success = True
    f0.subfactories[1].stage_1_success = True
    f1.subfactories[0].stage_1_success = False
    f1.subfactories[1].stage_1_success = False
    counts = {0: 2, 1: 0}
    transfers = redistribute_stage1_successes(pool, [0, 1], counts, None, 0.0, 0)
    assert counts == {0: 1, 1: 1}
    assert transfers == [(0, 1)]


def test_redistribute_no_op_when_all_have_at_least_one():
    pool = TFactoryPool(2, num_subfactories=2)
    pool.set_locations([(0, 0), (10, 0)])
    counts = {0: 1, 1: 1}
    transfers = redistribute_stage1_successes(pool, [0, 1], counts, None, 0.0, 0)
    assert transfers == []
    assert counts == {0: 1, 1: 1}
