"""
tissue_library.py:

Tissue parameter library and MP2RAGE protocol definitions.

Tissue T1 values come in two sets:
  - Adult values: canonical literature values
  - Paediatric values: estimated from real dataset histogram (inspect_t1map.py)

The active set used by run_simulation.py and flaws_dual_null.py is
TISSUE_PARAMS, which points to the paediatric values by default.
To switch back to adult, change the assignment at the bottom.

PD values are normalised (WM~0.69, GM~0.81, CSF~1.0) from literature.
"""


#  Adult VALUES (literature)

TISSUE_PARAMS_ADULT = {
    "white_matter": {"T1": 1092, "T2star": 28,   "PD": 0.69},
    "grey_matter":  {"T1": 1690, "T2star": 32.2, "PD": 0.81},
    "csf":          {"T1": 4470, "T2star": 200,  "PD": 1.0},
}

#  Paediatric VALUES  (estimated from Child T1 histogram)
#  WM T1 = 1095 ms  - close to adult, myelination nearly complete in WM
#  GM T1 = 1761 ms  - longer than adult, consistent with developing cortex
#  Source: inspect_t1map.py histogram peak detection on Child01 dataset
# ─────────────────────────────────────────────────────────────
TISSUE_PARAMS_PAEDIATRIC = {
    "white_matter": {"T1": 1048, "T2star": 28,   "PD": 0.69},
    "grey_matter":  {"T1": 1605, "T2star": 32.2, "PD": 0.81},
    "csf":          {"T1": 4470, "T2star": 200,  "PD": 1.0},
}

# Active tissue params (change this line to switch dataset)
TISSUE_PARAMS = TISSUE_PARAMS_PAEDIATRIC


#  MP2RAGE PROTOCOL

PROTOCOLS = {
    "protocol_1": {
        "TR_MP2RAGE": 5000,     # ms  — full repetition time
        "TI1": 650,             # ms  — centre of GRE1 block
        "TI2": 2220,            # ms  — centre of GRE2 block
        "alpha1": 5,            # degrees
        "alpha2": 7,            # degrees
        "TR_GRE": 7,            # ms  — GRE readout TR
        "TE": 3.0,              # ms
        "eff": 1.0,             # inversion efficiency
        "partitions": 320,
        "GRAPPA_factor": 1,
        "n": None               # computed below
    }
}

# n = excitations per GRE block (not total partitions)
p = PROTOCOLS["protocol_1"]
p["n"] = int(p["partitions"] / (2 * p["GRAPPA_factor"]))
