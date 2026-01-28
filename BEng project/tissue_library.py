"""
Tissue parameter library + baseline MP2RAGE protocol for 7T simulations.

Option A setup:
- We will sweep TI1.
- We will keep the spacing (TI2 - TI1) constant by updating TI2 inside run_simulation.py.
- n is set as excitations per GRE block (not total partitions).
"""

TISSUE_PARAMS = {
    "white_matter_adult": {"T1": 1092, "T2star": 28, "PD": 0.69},
    "grey_matter_adult":  {"T1": 1690, "T2star": 32.2, "PD": 0.81},
    "csf":               {"T1": 4470, "T2star": 200, "PD": 1.0},
}

PROTOCOLS = {
    "protocol_1": {
        "TR_MP2RAGE": 4000,     # ms
        "TI1": 650,             # ms (centre of GRE1)
        "TI2": 2220,            # ms (centre of GRE2)
        "alpha1": 5,            # degrees
        "alpha2": 4,            # degrees
        "TR_GRE": 7.0,          # ms
        "TE": 3.0,              # ms
        "eff": 1.0,             # inversion efficiency

        # NOTE:
        # n must be the number of excitations per GRE block.
        # A common first approximation for two readouts is partitions/2.
        "partitions": 320,
        "GRAPPA_factor": 1,
        "n": int(320 / (2 * 1)),  # 160
    }
}
