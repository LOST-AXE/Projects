"""
MP2RAGE signal simulator for quantitative MRI at 7T.

Implements MP2RAGE signal equations to simulate INV1 and INV2.

Notes:
- TI1 and TI2 are assumed to be to the *center* of each GRE readout block.
- n is the number of GRE excitations *per inversion block* (NOT total partitions).
"""

import numpy as np


class MP2RAGESimulator:
    """Simulates MP2RAGE signals and generates quantitative maps."""

    def __init__(self, protocol: dict, verbose: bool = False):
        # Sequence parameters
        self.TR_MP2RAGE = float(protocol["TR_MP2RAGE"])
        self.TI1 = float(protocol["TI1"])
        self.TI2 = float(protocol["TI2"])
        self.alpha1 = float(protocol["alpha1"])
        self.alpha2 = float(protocol["alpha2"])
        self.TR_GRE = float(protocol["TR_GRE"])
        self.n = int(protocol["n"])
        self.TE = float(protocol["TE"])
        self.eff = float(protocol.get("eff", 1.0))

        # Derived timing (ms)
        gre_block_ms = self.n * self.TR_GRE
        self.gre_block_ms = gre_block_ms

        self.TA = self.TI1 - gre_block_ms / 2.0
        self.TB = self.TI2 - self.TI1 - gre_block_ms
        self.TC = self.TR_MP2RAGE - (self.TI2 + gre_block_ms / 2.0)

        if verbose:
            print("MP2RAGE Simulator initialized:")
            print(f"  GRE block: n={self.n} × TR_GRE={self.TR_GRE}ms = {gre_block_ms:.1f}ms")
            print(f"  Timing: TA={self.TA:.1f}ms, TB={self.TB:.1f}ms, TC={self.TC:.1f}ms")

    @staticmethod
    def deg2rad(angle_deg: float) -> float:
        return float(np.deg2rad(angle_deg))

    def timing_is_valid(self) -> bool:
        """Physical validity check: all timing segments must be non-negative."""
        return (self.TA >= 0) and (self.TB >= 0) and (self.TC >= 0)

    def longitudinal_mag(self, T1: float, PD: float):
        """Calculate steady-state longitudinal magnetization."""
        alpha1 = self.deg2rad(self.alpha1)
        alpha2 = self.deg2rad(self.alpha2)

        EA = np.exp(-self.TA / T1)
        EB = np.exp(-self.TB / T1)
        EC = np.exp(-self.TC / T1)
        E1 = np.exp(-self.TR_GRE / T1)

        cos_a1_E1 = np.cos(alpha1) * E1
        cos_a2_E1 = np.cos(alpha2) * E1

        # Clean readable form of your numerator (same meaning, fewer bracket traps)
        termA = (1 - EA) * (cos_a1_E1 ** self.n) + (1 - E1) * (1 - (cos_a1_E1 ** self.n)) / (1 - cos_a1_E1)
        termB = termA * EB + (1 - EB)
        termC = termB * (cos_a2_E1 ** self.n) + (1 - E1) * (1 - (cos_a2_E1 ** self.n)) / (1 - cos_a2_E1)

        numerator = termC * PD * EC + (1 - EC) * PD

        denominator = 1 + self.eff * (np.cos(alpha1) * np.cos(alpha2)) ** self.n * np.exp(-self.TR_MP2RAGE / T1)

        m_zss = numerator / denominator
        return m_zss, EA, EB, EC, E1, cos_a1_E1, cos_a2_E1

    def calculate_signals(self, T1: float, PD: float, T2star: float = 30.0, B1minus: float = 1.0):
        """Calculate INV1 and INV2 signals for given tissue parameters."""
        if not self.timing_is_valid():
            raise ValueError(
                f"Invalid timing: TA={self.TA:.1f}, TB={self.TB:.1f}, TC={self.TC:.1f}. "
                "Fix TI1/TI2/TR_MP2RAGE/n/TR_GRE or constrain your sweep."
            )

        m_zss, EA, EB, EC, E1, cos_a1_E1, cos_a2_E1 = self.longitudinal_mag(T1, PD)

        alpha1 = self.deg2rad(self.alpha1)
        alpha2 = self.deg2rad(self.alpha2)

        decay = np.exp(-self.TE / T2star)

        # INV1
        INV1 = PD * B1minus * decay * np.sin(alpha1) * (
            (-self.eff * (m_zss / PD) * EA + (1 - EA)) * (cos_a1_E1 ** (self.n / 3.0))
            + (1 - E1) * (1 - (cos_a1_E1 ** (self.n / 3.0))) / (1 - np.cos(alpha1) * E1)
        )

        # INV2
        INV2 = PD * B1minus * decay * np.sin(alpha2) * (
            ((m_zss / PD) - (1 - EC)) / EC * (cos_a2_E1 ** (2.0 * self.n / 3.0))
            - (1 - E1) * (cos_a2_E1 ** (-2.0 * self.n / 3.0 - 1.0)) / (1 - np.cos(alpha2) * E1)
        )

        return INV1, INV2
