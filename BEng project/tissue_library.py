"""
Tissue parameter library for 7T MP2RAGE simulations.

This module provides tissue-specific MRI parameters (T1, T2*, PD) for
simulating MP2RAGE signals at 7 Tesla. Parameters are sourced from rough
logic - based estimates, peer-reviewed literature and Dr. Carmichael's
previous work.

Sources:
    - T1 and PD* values for White and gray matter:
        Dokumacı AS et al. Quantitative T1 and Effective Proton Density (PD*)
        mapping in children and adults at 7T. 2024
    - CSF T1 value at 7-T MRI:
        Bluestein KT et al. T1 and proton density at 7 T in patients with
         multiple sclerosis. Magn Reson Imaging. 2011;30(1):19-25.
         doi:10.1016/j.mri.2011.02.006.
    - T2* estimates for Gray matter at 7-T MRI:
        Cohen-Adad J et al. T2* mapping and B0 orientation-dependence at 7 T
        reveal cyto- and myeloarchitecture organization of the human cortex.
        Neuroimage. 2012;60(2):1006-14. doi:10.1016/j.neuroimage.2012.01.053.

    - Protocol parameters: From our MP2RAGE sequence optimization
"""

TISSUE_PARAMS = {
    'white_matter_adult': {
        'T1': 1092,      # Dokumacı et al., Quantitative T1 and Effective
                         # Proton Density.
        'T2star': 28,    # Educated guess
        'PD': 0.69       # Dokumacı et al., Quantitative T1 and Effective
                         # Proton Density
    },
    'grey_matter_adult': {
        'T1': 1690,      # Dokumacı et al., Quantitative T1 and Effective
                         # Proton Density
        'T2star': 32.2,  # Cohen-Adad et al., T2 mapping and B0
                         # orientation-dependence
        'PD': 0.81       # Dokumacı et al., Quantitative T1 and Effective
                         # Proton Density
    },

    'csf': {
        'T1': 4470,      # Bluestein et al., T1 and proton density at 7 T.
        'T2star': 200,   # Educated Guess
        'PD': 1.0        # Reference value
    }
}

# Protocol parameters from Dokumacı et al., Quantitative T1 and Effective
# Proton Density
PROTOCOLS = {
    'protocol_1': {
        'TR_MP2RAGE': 10000,    # (ms) - Total repetition time
        'TI1': 650,            # (ms) - First inversion time
        'TI2': 9000,           # (ms) - Second inversion time
        'alpha1': 5,           # (degrees) - First flip angle
        'alpha2': 1,           # (degrees) - Second flip angle
        'TE': 3.0,             # (ms) - Echo time (estimated)
        'partitions': 1,       # Number of slices in 3D volume
        'GRAPPA_factor': 1,    # Parallel imaging acceleration factor
        'TR_GRE': 7.0,         # (ms) - Time between excitations (estimated)
        'n': 1                 # Calculated: partitions / GRAPPA_factor = 1/1
    }
}