"""
Tissue parameter library + baseline MP2RAGE protocol for 7T simulations.

n is set as excitations per GRE block (not total partitions).
"""
from turtledemo.sorting_animate import partition

from jedi.inference.gradual.typing import Protocol

TISSUE_PARAMS = {
    "white_matter_adult": {"T1": 1092, "T2star": 28, "PD": 0.69},
    "grey_matter_adult":  {"T1": 1690, "T2star": 32.2, "PD": 0.81},
    "csf":               {"T1": 4470, "T2star": 200, "PD": 1.0},
}

PROTOCOLS = {
    "protocol_1": {
        "TR_MP2RAGE": 5000,     # ms
        "TI1": 650,             # ms (centre of GRE1)
        "TI2": 2220,            # ms (centre of GRE2)
        "alpha1": 5,            # degrees
        "alpha2": 7,            # degrees
        "TR_GRE": 7.0,          # ms
        "TE": 3.0,              # ms
        "eff": 1.0,             # inversion efficiency
        "partitions": 320,
        "GRAPPA_factor": 1,
        "n": None
    }
}
p = PROTOCOLS["protocol_1"]
p["n"] = int(p["partitions"] / (2 * p["GRAPPA_factor"]))


