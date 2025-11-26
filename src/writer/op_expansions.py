"""
ComboInst Expansion Functions.

These functions define how a ComboInst object is broken down
into a list of BaseInst or other ComboInst objects.
"""

from itertools import chain
from copy import deepcopy
from collections import defaultdict


from zac.writer.inst import RearrangeJob
from zac.writer.inst import OneQGate


def rearrange_rowbyrow(job: RearrangeJob):
    """Expand a high-level rearrange job into a sequence of detailed instructions that can be executed by the hardware.

    The expansion follows a row-by-row strategy, including detailed instructions:
    - Activation of rows and columns
    - Parking movements to avoid collisions
    - Big movements to new positions
    - Deactivation of rows and columns

    Parameters
    ----------
        - job: RearrangeJob
    """
    inst = job.prompt
    details = []  # all detailed instructions

    # ---------------------- find out number of cols ----------------------
    all_col_x = []  # all the x coord of qubits
    coords = []  # coords of qubits, shape is same as "begin_locs"
    # these coords are going to be updated as we construct the detail insts

    for locs in inst["begin_locs"]:
        coords_row = []
        for loc in locs:
            exact_location = job.architecture.exact_SLM_location(loc[1], loc[2], loc[3])
            coords_row.append(
                {
                    "id": loc[0],
                    "x": exact_location[0],
                    "y": exact_location[1],
                }
            )

            all_col_x.append(exact_location[0])

        coords.append(coords_row)

    init_coords = deepcopy(coords)

    all_col_x = sorted(all_col_x)

    # assign AOD column ids based on all x coords needed
    col_x_to_id = {all_col_x[i]: i for i in range(len(all_col_x))}
    # ---------------------------------------------------------------------

    # -------------------- activation and parking -------------------------
    all_col_idx_sofar = []  # which col has been activated
    for row_id, locs in enumerate(inst["begin_locs"]):  # each row

        row_y = job.architecture.exact_SLM_location(
            locs[0][1],
            locs[0][2],
            locs[0][3],
        )[1]
        row_loc = [locs[0][1], locs[0][2]]

        # before activation, adjust column position. This is necessary
        # whenever cols are parked (the `parking` movement below).
        shift_back = {
            "type": "move",
            "move_type": "before",
            "row_id": [],
            "row_y_begin": [],
            "row_y_end": [],
            "row_loc_begin": [],
            "row_loc_end": [],
            "col_id": [],
            "col_x_begin": [],
            "col_x_end": [],
            "col_loc_begin": [],
            "col_loc_end": [],
            "begin_coord": deepcopy(coords),
            "end_coord": [],
        }

        # activate one row and some columns
        activate = {
            "type": "activate",
            "row_id": [
                row_id,
            ],
            "row_y": [
                row_y,
            ],
            "row_loc": [
                row_loc,
            ],
            "col_id": [],
            "col_x": [],
            "col_loc": [],
        }

        for j, loc in enumerate(locs):
            col_x = job.architecture.exact_SLM_location(
                loc[1],
                loc[2],
                loc[3],
            )[0]
            col_loc = [loc[1], loc[3]]
            col_id = col_x_to_id[col_x]
            if col_id not in all_col_idx_sofar:
                # the col hasn't been activated, so there's no shift back
                # and we need to activate it at `col_x`.`
                all_col_idx_sofar.append(col_id)
                activate["col_id"].append(col_id)
                activate["col_x"].append(col_x)
                activate["col_loc"].append(col_loc)
            else:
                # the col has been activated, thus parked previously and we
                # need the shift back, but we do not activate again.
                shift_back["col_id"].append(col_id)
                shift_back["col_x_begin"].append(col_x + job.PARKING_DIST)
                shift_back["col_x_end"].append(col_x)
                shift_back["col_loc_begin"].append([-1, -1])
                shift_back["col_loc_end"].append(col_loc)
                # since there's a shift, update the coords of the qubit
                coords[row_id][j]["x"] = col_x

        shift_back["end_coord"] = deepcopy(coords)

        if len(shift_back["col_id"]) != 0:
            details.append(shift_back)
        details.append(activate)

        if row_id < len(inst["begin_locs"]) - 1:
            # parking movement after the activation
            # parking is required if we have activated some col, and there is
            # some qubit we don't want to pick up at the intersection of this
            # col and some future row to activate. We just always park here.
            # the last parking is not needed since there's a big move after it.
            parking = {
                "type": "move",
                "move_type": "after",
                "row_id": [
                    row_id,
                ],
                "row_y_begin": [
                    row_y,
                ],
                "row_y_end": [row_y + job.PARKING_DIST],
                "row_loc_begin": [row_loc],
                "row_loc_end": [[-1, -1]],
                "col_id": [],
                "col_x_begin": [],
                "col_x_end": [],
                "col_loc_begin": [],
                "col_loc_end": [],
                "begin_coord": deepcopy(coords),
                "end_coord": [],
            }
            for j, loc in enumerate(locs):
                col_x = job.architecture.exact_SLM_location(
                    loc[1],
                    loc[2],
                    loc[3],
                )[0]
                col_loc = [loc[1], loc[3]]
                col_id = col_x_to_id[col_x]
                # all columns used in this row are parked after the activation
                parking["col_id"].append(col_id)
                parking["col_x_begin"].append(col_x)
                parking["col_x_end"].append(col_x + job.PARKING_DIST)
                parking["col_loc_begin"].append(col_loc)
                parking["col_loc_end"].append([-1, -1])
                coords[row_id][j]["x"] = parking["col_x_end"][-1]
                coords[row_id][j]["y"] = parking["row_y_end"][0]
            parking["end_coord"] = deepcopy(coords)
            details.append(parking)
    # ---------------------------------------------------------------------

    # ------------------------- big move ----------------------------------
    big_move = {
        "type": "move:big",
        "move_type": "big",
        "row_id": [],
        "row_y_begin": [],
        "row_y_end": [],
        "row_loc_begin": [],
        "row_loc_end": [],
        "col_id": [],
        "col_x_begin": [],
        "col_x_end": [],
        "col_loc_begin": [],
        "col_loc_end": [],
        "begin_coord": deepcopy(coords),
        "end_coord": [],
    }

    for row_id, (begin_locs, end_locs) in enumerate(
        zip(
            inst["begin_locs"],
            inst["end_locs"],
        )
    ):

        big_move["row_id"].append(row_id)
        big_move["row_y_begin"].append(coords[row_id][0]["y"])
        if init_coords[row_id][0]["y"] == coords[row_id][0]["y"]:
            # AOD row is align with SLM row
            big_move["row_loc_begin"].append([begin_locs[0][1], begin_locs[0][2]])
        else:
            big_move["row_loc_begin"].append([-1, -1])

        big_move["row_y_end"].append(
            job.architecture.exact_SLM_location(
                end_locs[0][1],
                end_locs[0][2],
                end_locs[0][3],
            )[1]
        )
        big_move["row_loc_end"].append([end_locs[0][1], end_locs[0][2]])

        for j, (begin_loc, end_loc) in enumerate(zip(begin_locs, end_locs)):
            col_x = job.architecture.exact_SLM_location(
                begin_loc[1],
                begin_loc[2],
                begin_loc[3],
            )[0]
            col_id = col_x_to_id[col_x]

            if col_id not in big_move["col_id"]:
                # the movement of this rol has not been recorded before
                big_move["col_id"].append(col_id)
                big_move["col_x_begin"].append(coords[row_id][j]["x"])
                if init_coords[row_id][j]["x"] == coords[row_id][j]["x"]:
                    # AOD col is align with SLM col
                    big_move["col_loc_begin"].append([begin_loc[1], begin_loc[3]])
                else:
                    big_move["col_loc_begin"].append([-1, -1])

                big_move["col_x_end"].append(
                    job.architecture.exact_SLM_location(
                        end_loc[1],
                        end_loc[2],
                        end_loc[3],
                    )[0]
                )
                big_move["col_loc_end"].append([end_loc[1], end_loc[3]])

            # whether or not the movement of this col has been considered
            # before, we need to update the coords of the qubit.

            coords[row_id][j]["x"] = job.architecture.exact_SLM_location(
                end_loc[1],
                end_loc[2],
                end_loc[3],
            )[0]

            coords[row_id][j]["y"] = job.architecture.exact_SLM_location(
                end_locs[0][1],
                end_locs[0][2],
                end_locs[0][3],
            )[1]

    big_move["end_coord"] = deepcopy(coords)
    details.append(big_move)
    # ---------------------------------------------------------------------

    # --------------------------- deactivation ----------------------------
    details.append(
        {
            "type": "deactivate",
            "row_id": [i for i in range(len(inst["begin_locs"]))],
            "col_id": [i for i in range(len(all_col_x))],
        }
    )
    # ---------------------------------------------------------------------

    for inst_counter, detail_inst in enumerate(details):
        detail_inst["id"] = inst_counter

    # Adding Details and Flatten Rearrange
    job.code = job.prompt
    job.code["insts"] = details
    job.code["aod_qubits"] = list(chain.from_iterable(job.code["aod_qubits"]))
    job.code["begin_locs"] = list(chain.from_iterable(job.code["begin_locs"]))
    job.code["end_locs"] = list(chain.from_iterable(job.code["end_locs"]))


def coalesce_same_angle(single_qubit_gates: OneQGate):
    """Coalesce consecutive single-qubit gates with the same angle into a single gate instruction.

    It modifies the `inst` attribute of the
    `OneQGate` object in place. Following the ASAP principle.

    Parameters
    ----------
    single_qubit_gates: OneQGate :
    """
    inst_original = single_qubit_gates.code["inst"]
    inst_coalesced: list[dict] = []
    inst_dependency_dict = defaultdict(int)  # loc_id -> index in inst_coalesced

    for gate in inst_original:
        angle = gate["angle"]
        locs = gate["locs"]
        loc_id = locs[0][0]

        dependent_idx = inst_dependency_dict.get(loc_id)
        if dependent_idx is None:
            dependent_idx = 0

        flag_found = False
        for inst in inst_coalesced[dependent_idx:]:
            if inst["angle"] == angle:
                inst["locs"].append(locs[0])
                inst_dependency_dict[loc_id] = inst_coalesced.index(inst)
                flag_found = True
                break
        if not flag_found:
            inst_coalesced.append({"angle": angle, "locs": [locs[0]]})
            inst_dependency_dict[loc_id] = len(inst_coalesced) - 1

    single_qubit_gates.code["inst"] = inst_coalesced
    # print(
    #     f"Coalesced {len(inst_original)} single-qubit gates into {len(inst_coalesced)} gates."
    # )


DEFAULT_EXPANSION_STRATEGIES = {
    "rearrangeJob": rearrange_rowbyrow,
    "1qGate": coalesce_same_angle,
}
