"""
Architecture data structures and utilities.

Provides classes and helpers modeling the device geometry and timing used by
ZAC scheduler and router:

- AOD: Acoustic-optic deflector array specification.
- SLM: Spatial light modulator (storage/entanglement) specification.
- Architecture: Top-level device topology, site lookups, distance and timing
  helpers (movement_duration, nearest entanglement/site lookups, preprocessing).
"""

import math


class AOD:
    """Class to AOD array."""

    def __init__(self, aod_spec: dict):
        """Initialize AOD with specifications from a dictionary.

        Args
        ------
        - `aod_spec` : dict
            A dictionary containing the specifications for the AOD.
            Expected keys:
            - "id": int, the identifier for the AOD.
            - "site_seperation": int, the separation between sites in the AOD.
            - "r": int, the number of rows in the AOD.
            - "c": int, the number of columns in the AOD.
        """
        self.idx = -1
        self.site_seperation = 0
        self.n_r = 0
        self.n_c = 0
        if "id" in aod_spec:
            self.idx = aod_spec["id"]
        else:
            raise ValueError("AOD id is missed in architecture spec")
        if "site_seperation" in aod_spec:
            self.site_seperation = aod_spec["site_seperation"]
        else:
            raise ValueError("AOD site seperation is missed in architecture spec")
        if "r" in aod_spec:
            self.n_r = aod_spec["r"]
        else:
            raise ValueError("AOD row number is missed in architecture spec")
        if "c" in aod_spec:
            self.n_c = aod_spec["c"]
        else:
            raise ValueError("AOD column number is missed in architecture spec")


class SLM:
    """Class to SLM array."""

    def __init__(self, slm_spec: dict):
        """Initialize SLM with specifications from a dictionary.

        Args
        ------
        - `slm_spec` : dict
            A dictionary containing the specifications for the SLM.
            Expected keys:
            - "id": int, the identifier for the SLM.
            - "site_seperation": list of two ints, the separation between sites in x and y directions.
            - "r": int, the number of rows in the SLM.
            - "c": int, the number of columns in the SLM.
            - "location": list of two ints, the (x, y) location of the SLM.
        """
        self.idx = -1
        self.site_seperation: list[int] = [0, 0]
        self.n_r: int = 0
        self.n_c: int = 0
        self.location: list[int] = []
        self.entanglement_id: int = -1
        if "id" in slm_spec:
            self.idx = slm_spec["id"]
        else:
            raise ValueError("SLM id is missed in architecture spec")
        if "site_seperation" in slm_spec:
            self.site_seperation = slm_spec["site_seperation"]
        else:
            raise ValueError("SLM site seperation is missed in architecture spec")
        if "r" in slm_spec:
            self.n_r = slm_spec["r"]
        else:
            raise ValueError("SLM row number is missed in architecture spec")
        if "c" in slm_spec:
            self.n_c = slm_spec["c"]
        else:
            raise ValueError("SLM column number is missed in architecture spec")
        if "location" in slm_spec:
            self.location = slm_spec["location"]
        else:
            raise ValueError("SLM location is missed in architecture spec")


class Architecture:
    """Class to define zone architecture."""

    def __init__(self, architecture_spec: dict):
        """Initialize Architecture with specifications from a dictionary.

        Parameters
        ----------
        - `architecture_spec` : dict
            A dictionary containing the specifications for the architecture.
            Expected keys:
            - "name": str, the name of the architecture.
            - "operation_duration": dict, the duration of operations in the architecture.
            - "storage_zones": list of dicts, the storage zones in the architecture.
            - "entanglement_zones": list of dicts, the entanglement zones in the architecture.
            - "aods": list of dicts, the AODs in the architecture.
        """
        self.name = None
        self.operation_duration = dict()
        self.storage_zone = []
        self.entanglement_zone = []
        self.dict_SLM: dict[int, SLM] = dict()
        self.dict_AOD: dict[int, AOD] = dict()
        self.time_atom_transfer = 15  # us
        self.time_rydberg = 0.36  # us
        self.time_1qGate = 0.625  # us
        # parse architecture name
        if "name" in architecture_spec:
            self.name = architecture_spec["name"]
        # parse architecture operation duration
        if "operation_duration" in architecture_spec:
            if "rydberg" in architecture_spec["operation_duration"]:
                self.operation_duration["rydberg"] = architecture_spec[
                    "operation_duration"
                ]["rydberg"]
                self.time_rydberg = architecture_spec["operation_duration"]["rydberg"]
            if "atom_transfer" in architecture_spec["operation_duration"]:
                self.operation_duration["atom_transfer"] = architecture_spec[
                    "operation_duration"
                ]["atom_transfer"]
                self.time_atom_transfer = architecture_spec["operation_duration"][
                    "atom_transfer"
                ]
            if "1qGate" in architecture_spec["operation_duration"]:
                self.operation_duration["1qGate"] = architecture_spec[
                    "operation_duration"
                ]["1qGate"]
                self.time_1qGate = architecture_spec["operation_duration"]["1qGate"]

        # parse architecture zone information
        if "arch_range" in architecture_spec:
            self.arch_range = architecture_spec["arch_range"]
        if "rydberg_range" in architecture_spec:
            self.rydberg_range = architecture_spec["rydberg_range"]
        if "storage_zones" in architecture_spec:
            for zone in architecture_spec["storage_zones"]:
                for slm_spec in zone["slms"]:
                    slm = SLM(slm_spec)
                    self.dict_SLM[slm.idx] = slm
                    self.storage_zone.append(slm.idx)
        if "entanglement_zones" in architecture_spec:
            y_slm = dict()
            for zone in architecture_spec["entanglement_zones"]:
                for slm_spec in zone["slms"]:
                    slm = SLM(slm_spec)
                    self.dict_SLM[slm.idx] = slm
                    if slm.location[1] in y_slm:
                        self.entanglement_zone[y_slm[slm.location[1]]].append(slm.idx)
                    else:
                        y_slm[slm.location[1]] = len(self.entanglement_zone)
                        self.entanglement_zone.append([slm.idx])
                    slm.entanglement_id = zone["zone_id"]
        else:
            raise ValueError(
                "entanglement zone configuration is missed in architecture spec"
            )

        # parse AOD information
        if "aods" in architecture_spec:
            for aod_spec in architecture_spec["aods"]:
                aod = AOD(aod_spec)
                self.dict_AOD[aod.idx] = aod
        else:
            raise ValueError("AOD is missed in architecture spec")

    def is_valid_SLM(self, idx: int) -> bool:
        """Check if the given SLM index is valid.

        Parameters
        ----------
        - `idx` : int
            The index of the SLM to check.
        """
        return idx in self.dict_SLM

    def is_valid_SLM_position(self, idx: int, r: int, c: int) -> bool:
        """Check if the given position (r, c) is valid for the specified SLM index.

        Parameters
        -------
        - `idx` : int
            The index of the SLM to check.
        - `r` : int
            The row position to check.
        - `c` : int
            The column position to check.
        """
        return (
            r < self.dict_SLM[idx].n_r
            and c < self.dict_SLM[idx].n_c
            and r >= 0
            and c >= 0
        )

    def is_valid_AOD(self, idx) -> bool:
        """Check if the given AOD index is valid.

        Parameters
        ------
        - `idx` : int
            The index of the AOD to check.
        """
        return idx in self.dict_AOD

    def n_AOD(self) -> int:
        """Get the number of AODs."""
        return len(self.dict_AOD)

    def exact_SLM_location(self, idx: int, r: int, c: int) -> tuple[int, int]:
        """Get the exact (x, y) location of a site in the specified SLM.

        Parameters
        ------
        - `idx` : int
            The index of the SLM.
        - `r` : int
            The row position of the site.
        - `c` : int
            The column position of the site.

        Returns
        -------
        - tuple of two ints
            The (x, y) location of the site.
        """
        slm = self.dict_SLM[idx]
        assert self.is_valid_SLM_position(idx, r, c)
        x = slm.site_seperation[0] * c + slm.location[0]
        y = slm.site_seperation[1] * r + slm.location[1]
        return (x, y)

    def exact_SLM_location_tuple(self, loc: tuple[int, int, int]) -> tuple[int, int]:
        """Get the exact (x, y) location of a site in the specified SLM.

        Parameters
        ------
        - `loc` : tuple of three ints
            The (SLM index, row position, column position) of the site.

        Returns
        -------
        - tuple of two ints
            The (x, y) location of the site.
        """
        slm = self.dict_SLM[loc[0]]
        assert self.is_valid_SLM_position(loc[0], loc[1], loc[2])
        x = slm.site_seperation[0] * loc[2] + slm.location[0]
        y = slm.site_seperation[1] * loc[1] + slm.location[1]
        return (x, y)

    def movement_duration(self, x1, y1, x2, y2) -> float:
        """Estimate movement time to move between two (x, y) positions.

        The implementation uses a simple kinematic approximation:
        t = sqrt(d / a)
        where d is the Euclidean distance between (x1, y1) and (x2, y2),
        and `a` is a constant acceleration-like parameter.

        Parameters
        ------
        - `x1`, `y1` : float
            Coordinates of the start position.
        - `x2`, `y2` : float
            Coordinates of the target position.

        Returns
        -------
        - float
            Estimated time to move between the two positions using the
            same distance units as the input coordinates. The time units
            correspond to the units implied by the constant `a` used below.

        Notes
        -----
        - `a` is a tuned constant (0.00275 in this code). Ensure `a` is
          consistent with the units of the provided coordinates.
        - This is a simple model and does not account for acceleration/deceleration
          profiles, maximum velocity caps, or hardware-specific constraints.
        """
        a = 0.00275
        d = math.dist((x1, y1), (x2, y2))
        # d= 15
        t = math.sqrt(d / a)
        return t


def move_duration(x1, y1, x2, y2) -> float:
    movement_time = abs(x1 - x2) + abs(y1 - y2)
    movement_time /= 2
    return movement_time
