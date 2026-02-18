from src.rus.rus_routing import solve_return_move
from src.ds import FactoryPool


def rus_post_teleportation(
    routing_batches: list[list[tuple[int, int, int, int, int, int]]],
    factory_pool: FactoryPool,
    trivial_return: bool = False,
    decompose_move: bool = False,
) -> list[list[tuple[int, int, int, int, int, int]]]:
    # return factories qubit to empty spot
    if trivial_return:
        return_routing_batches = []
        for batches in routing_batches:
            return_routing_batches.append([])
            for _, x_q, y_q, factory_id, x_f, y_f in batches:
                return_routing_batches[-1].append((-1, x_f, y_f, factory_id, x_q, y_q))
    else:
        return_routing_batches = solve_return_move(
            routing_batches, factory_pool, decompose_move=decompose_move
        )
    return return_routing_batches
