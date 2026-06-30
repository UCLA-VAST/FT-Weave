import math

from src.circuit.rz_params import (
    combine_rz_angles,
    normalize_rz_angle,
    rz_params_from_angles,
    rz_target_angles,
)


def test_rz_target_angles_from_theta():
    instr = {"gate": "Rz", "targets": [0, 1], "params": {"theta": 0.5}}
    assert rz_target_angles(instr) == {0: 0.5, 1: 0.5}


def test_rz_target_angles_from_angles_dict():
    instr = {
        "gate": "Rz",
        "targets": [0, 1],
        "params": {"angles": {0: 0.5, 1: 0.3}},
    }
    assert rz_target_angles(instr) == {0: 0.5, 1: 0.3}


def test_combine_rz_angles_sums_and_normalizes():
    merged = combine_rz_angles({0: 0.5}, {0: math.pi})
    assert math.isclose(merged[0], normalize_rz_angle(0.5 + math.pi))


def test_rz_params_from_angles_uniform():
    params = rz_params_from_angles({0: 0.2, 1: 0.2})
    assert params["theta"] == 0.2
    assert params["angles"] == {0: 0.2, 1: 0.2}
