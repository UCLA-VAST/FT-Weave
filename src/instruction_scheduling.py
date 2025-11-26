from src.ds.architecture import Architecture


def instruction_scheduling(
    circuit_instructions: list[dict], architecture: Architecture
):
    # assum sequential execution for now
    time_rydberg = architecture.time_rydberg
    time_1qGate = architecture.time_1qGate
    circuit_time = 0
    for inst in circuit_instructions:
        inst["begin_time"] = circuit_time

        if inst["type"] == "rydberg":
            circuit_time += time_rydberg
        elif inst["type"] == "rearrangeJob":
            circuit_time = get_rearrangeJob_duration(circuit_time, inst, architecture)
        elif inst["type"] == "1qGate":
            circuit_time += time_1qGate
        elif not inst["type"] == "init":
            raise NotImplementedError
        inst["end_time"] = circuit_time


def get_rearrangeJob_duration(
    start_time: float, inst: dict, architecture: Architecture
) -> float:
    """Get the total duration of the instruction including all its sub-instructions.

    Parameters
    ------
    - `architecture` : Architecture
    - `inst` : dict
        Instruction dictionary containing a list of sub-instructions under the
        "insts" key. Each sub-instruction is expected to include keys such as
        "type", and for "move" entries the lists "row_y_begin", "row_y_end",
        "col_x_begin", "col_x_end".

    Returns
    -------
    - float
        Total duration of the instruction (in the same time units used by the
        Architecture timing parameters).

    Notes
    -----
    - This function updates each detail_inst with "begin_time" and "end_time"
      relative to the start of the instruction.
    - A ValueError is raised for unknown detail instruction types.
    """
    list_detail_inst = inst["insts"]
    end_time = start_time
    for detail_inst in list_detail_inst:
        inst_type = detail_inst["type"].split(":")[0]
        detail_inst["begin_time"] = end_time
        if inst_type == "activate" or inst_type == "deactivate":
            end_time += architecture.time_atom_transfer
            detail_inst["end_time"] = end_time
        elif inst_type == "move":
            move_duration = 0
            for row_begin, row_end in zip(
                detail_inst["row_y_begin"], detail_inst["row_y_end"]
            ):
                for col_begin, col_end in zip(
                    detail_inst["col_x_begin"], detail_inst["col_x_end"]
                ):
                    tmp = architecture.movement_duration(
                        col_begin, row_begin, col_end, row_end
                    )
                    if move_duration < tmp:
                        move_duration = tmp
            detail_inst["end_time"] = move_duration + end_time
            end_time += move_duration
        else:
            raise ValueError
    return end_time
