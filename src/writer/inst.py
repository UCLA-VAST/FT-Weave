"""
Instruction writer helpers for ZAC.

This module contains helper functions and small writer classes that convert
internal instruction representations (typically dicts produced by the
scheduler) into a serializable output form or device-specific command lines.
"""

from abc import ABC, abstractmethod
from typing import Sequence


class Inst(ABC):
    """Abstract class of DPQA instructions.

    In general, the __init__ of specific instruction classes looks like
    def __init__(self, *):         super().__init__(*)
    self.verify(*)         self.operate(*)         self.write(*) # for
    BaseInst only         self.prep(*)  # for ComboInst only
    """

    def __init__(
        self,
        type: str,
        prefix=None,
        stage: int = -1,
        reduced_keys: Sequence[str] = [],
    ):
        """Init method for instructions.

        Args:
            - type (str):
                prefix (str | None, optional): provide the big operation.
                    this Inst belongs to. Defaults to None.
            - stage (int, optional): stage the Inst belongs to. Defaults to -1.
            - reduced_keys (Sequence[str], optional): data to keep in emit()
                  from emit_full(). Defaults to [].
        """
        self.type = type
        self.name = prefix + ":" + type if prefix else type
        self.stage = stage
        self.reduced_keys = list(reduced_keys) + [
            "type",
        ]
        self.duration = -1
        self.code = {
            "type": self.type,
            "name": self.name,
        }

    @abstractmethod
    def verify(self):
        """Verification of instructions.

        This is abstract because we require each child class to provide
        its own verification method.
        """
        pass

    def operate(self):
        """Perform operation of instructions on Col, Row, and Qubit objects."""
        pass


class BaseInst(Inst):
    """ """

    def __init__(self, basetype: str):
        super().__init__(basetype)

    @abstractmethod
    def write(self):
        """Write self.code with provided info."""
        pass


class ComboInst(Inst):
    """Class of Combo Instruction.

    Expansion to Fine-Grained Instruction Needed.
    """

    def __init__(self, combotype: str):
        super().__init__(combotype)

    @abstractmethod
    def prep(self):
        """Prepare before expansion."""
        pass


class Init(BaseInst):
    """ """

    def __init__(self, prompt, qubit_mapping, n_q):
        super().__init__("init")

        self.prompt = prompt
        self.qubit_mapping = qubit_mapping
        self.n_q = n_q

        self.verify()
        self.operate()
        self.write()

    def write(self):
        """ """
        self.code = {
            "type": self.prompt["type"],
            "id": self.prompt["id"],
            "init_locs": [
                [
                    i,
                    self.qubit_mapping[0][i][0],
                    self.qubit_mapping[0][i][1],
                    self.qubit_mapping[0][i][2],
                ]
                for i in range(self.n_q)
            ],
        }

    def verify(self):
        """ """
        pass

    def operate(self):
        """ """
        pass


class OneQGate(ComboInst):
    """Class of 1qGate Instruction.

    Expansion to Fine-Grained Instruction Needed.
    """

    def __init__(self, prompt):
        super().__init__("1qGate")
        self.prompt = prompt

        self.verify()
        self.operate()
        self.prep()
        # external call to expansion function is needed

    def prep(self):
        """ """
        result_gate = self.prompt["gates"]
        gate_mapping = self.prompt["mapping"]

        details = []
        for gate in result_gate:
            angle = gate["angle"]
            loc = [
                [
                    gate["q"],
                    gate_mapping[gate["q"]][0],
                    gate_mapping[gate["q"]][1],
                    gate_mapping[gate["q"]][2],
                ]
            ]
            details.append({"angle": angle, "locs": loc})

        self.code = {
            "type": self.prompt["type"],
            "id": self.prompt["id"],
            "unitary": self.prompt["unitary"],
            "inst": details,
            "dependency": self.prompt["dependency"],
        }

    def verify(self):
        """ """
        pass

    def operate(self):
        """ """
        pass


class Rydberg(BaseInst):
    """ """

    def __init__(self, prompt):
        super().__init__("rydberg")
        self.prompt = prompt

        self.verify()
        self.operate()
        self.write()

    def write(self):
        """ """
        self.code = {
            "type": self.prompt["type"],
            "id": self.prompt["id"],
            "zone_id": self.prompt["zone_id"],
            "gates": self.prompt["gates"],
            "dependency": self.prompt["dependency"],
        }

    def verify(self):
        """ """
        pass

    def operate(self):
        """ """
        pass


class Activate(BaseInst):
    """ """

    def __init__(self):
        super().__init__("activate")
        self.verify()
        self.operate()
        self.write()

    def write(self):
        """ """
        pass

    def verify(self):
        """ """
        pass

    def operate(self):
        """ """
        pass


class Deactivate(BaseInst):
    """ """

    def __init__(self):
        super().__init__("deactivate")
        self.verify()
        self.operate()
        self.write()

    def write(self):
        """ """
        pass

    def verify(self):
        """ """
        pass

    def operate(self):
        """ """
        pass


class Move(BaseInst):
    """ """

    def __init__(self):
        super().__init__("move")
        self.verify()
        self.operate()
        self.write()

    def write(self):
        """ """
        pass

    def verify(self):
        """ """
        pass

    def operate(self):
        """ """
        pass


class RearrangeJob(ComboInst):
    """Class of rearrangeJob Instruction.

    Expansion to Fine-Grained Instruction Needed.
    """

    def __init__(self, prompt, architecture, parking_dist):
        """Class of rearrangeJob Instruction.

        Expansion to Fine-Grained Instruction Needed.

        Parameters
        ----------
            - prompt (dict): prompt dictionary containing the details of the instruction.
            - architecture: Architecture object
            - parking_dist (int): the distance of AOD row and col to some trap.
        """
        super().__init__("rearrangeJob")

        self.prompt = prompt
        self.architecture = architecture
        self.PARKING_DIST = parking_dist

        self.verify()
        self.operate()
        self.prep()
        # external call to expansion function is needed

    def prep(self):
        """ """
        pass

    def verify(self):
        """ """
        pass

    def operate(self):
        """ """
        pass


INSTRUCTIONS = {
    "init": Init,
    "1qGate": OneQGate,
    "rydberg": Rydberg,
    "activate": Activate,
    "deactivate": Deactivate,
    "move": Move,
    "rearrangeJob": RearrangeJob,
}

ARGUMENTS = {
    "init": ["qubit_mapping", "n_q"],
    "1qGate": [],
    "rydberg": [],
    "rearrangeJob": ["architecture", "parking_dist"],
}
