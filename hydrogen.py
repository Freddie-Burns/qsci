"""
Examples
--------
# Evenly spaced linear chain along z
>>> cfg = EvenlySpacedLinearHChainConfig(n_linear_atoms=6, bond_length=0.75)
>>> res = calculate_hydrogen_chain_energies(cfg)
>>> res.fci_energy
-3.14

# Arbitrary H-only arrangement
>>> cfg = HydrogenChainConfig(geometry="H 0 0 0; H 0.0 0.0 0.75; H 0.2 0.1 1.50")
>>> res = calculate_hydrogen_chain_energies(cfg)
>>> res.rhf_energy  # doctest: +ELLIPSIS
-...
"""

import math

from pydantic import BaseModel, Field, computed_field
from pyscf import gto, scf, fci

from molecule import Basis


# ---------------------------
# Base (general) configuration
# ---------------------------


class HydrogenChainConfig(BaseModel):
    """
    General configuration for an H-only system with an arbitrary geometry.

    Parameters
    ----------
    geometry : str
        PySCF-compatible geometry string (e.g., "H x y z; H x y z; ...").
    basis : Basis, default=Basis.STO_3G
        Basis set to use.
    charge : int, default=0
        Net molecular charge.
    verbose : int, default=1
        PySCF verbosity.

    Attributes
    ----------
    n_atoms : int
        Number of H atoms, inferred from `geometry`.
    spin : int
        Number of unpaired electrons, computed as (n_atoms - charge) % 2.
    """
    geometry: str = Field(..., description="PySCF geometry for H-only system")
    basis: Basis = Field(default=Basis.STO_3G)
    charge: int = 0
    verbose: int = 1

    @computed_field(return_type=int)
    def n_atoms(self) -> int:
        """Number of H atoms inferred from the geometry string."""
        # split on ';' while tolerating extra whitespace
        atoms = [a.strip() for a in self.geometry.split(";") if a.strip()]
        return len(atoms)

    @property
    def spin(self) -> int:
        """Number of unpaired electrons; 0 indicates a singlet state."""
        return (self.n_atoms - self.charge) % 2


# -----------------------------------------------------
# Specialized config: evenly spaced linear H chain (z)
# -----------------------------------------------------


class EvenlySpacedLinearHChainConfig(HydrogenChainConfig):
    """
    Specialized configuration for an evenly spaced linear hydrogen chain
    aligned along the z-axis.

    Parameters
    ----------
    n_atoms : int
        Number of H atoms in the chain (>= 1).
    bond_length : float
        Distance between neighboring H atoms in angstroms (> 0).

    Notes
    -----
    This subclass *defines* `geometry` from `n_linear_atoms` and `bond_length`.
    Other attributes (basis, charge, verbose) are inherited.
    """
    # n_atoms is overriding the computed_field from the base class.
    n_atoms: int = Field(..., ge=1, description="Number of H atoms")
    bond_length: float = Field(..., gt=0, description="Å spacing along z")

    @computed_field(return_type=str)  # overrides base geometry
    def geometry(self) -> str:  # type: ignore[override]
        """PySCF geometry string for an evenly spaced chain along z."""
        return "; ".join(
            f"H 0 0 {n * self.bond_length:.6f}" for n in range(self.n_linear_atoms)
        )


# -------------
# Result object
# -------------


class HChainResult(BaseModel):
    """
    Result model for electronic structure calculations on a hydrogen chain.

    Stores total electronic energies from FCI, RHF, and UHF, plus the final
    molecular geometry specification.

    Parameters
    ----------
    fci_energy : float
        FCI total energy (Hartree).
    rhf_energy : float
        RHF total energy (Hartree).
    uhf_energy : float
        UHF total energy (Hartree).
    geometry : str
        PySCF-compatible geometry string.
    """
    fci_energy: float
    """FCI total energy in Hartree."""

    rhf_energy: float
    """RHF total energy in Hartree."""

    uhf_energy: float
    """UHF total energy in Hartree."""

    geometry: str
    """Geometry string in PySCF format."""

    # Helpful computed fields that work for arbitrary 3D arrangements
    @computed_field(return_type=list[tuple[float, float, float]])
    def positions(self) -> list[tuple[float, float, float]]:
        """
        Parsed (x, y, z) positions (Å) from the geometry string.
        """
        pts: list[tuple[float, float, float]] = []
        for atom in [a.strip() for a in self.geometry.split(";") if a.strip()]:
            parts = atom.split()
            # Expect "H x y z"
            if len(parts) != 4 or parts[0].upper() != "H":
                raise ValueError(
                    f"Invalid atom entry for hydrogen geometry: '{atom}'"
                )
            x, y, z = map(float, parts[1:])
            pts.append((x, y, z))
        return pts

    @computed_field(return_type=float)
    def first_bond_length(self) -> float:
        """
        Distance (Å) between the first two hydrogens (if >= 2 atoms),
        otherwise 0.0.
        """
        if len(self.positions) < 2:
            return 0.0
        (x1, y1, z1), (x2, y2, z2) = self.positions[0], self.positions[1]
        return float(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2))


# -----------------------
# Calculation entry point
# -----------------------


def calculate_hydrogen_chain_energies(config: HydrogenChainConfig) -> HChainResult:
    """
    Build and compute the electronic structure of an H-only system.

    Accepts either a general `HydrogenChainConfig` (arbitrary geometry)
    or the specialized `EvenlySpacedLinearHChainConfig` (equispaced z-chain).

    Parameters
    ----------
    config : HydrogenChainConfig
        Configuration object carrying geometry, basis, charge, and verbosity.

    Returns
    -------
    HChainResult
        RHF, UHF, and FCI total energies (Hartree) and the geometry string.

    Notes
    -----
    - Energies are returned in atomic units (Hartree).
    - Spin is inferred as (n_atoms - charge) % 2.
    """
    mol = gto.Mole()
    mol.atom = config.geometry
    mol.basis = config.basis
    mol.charge = config.charge
    mol.spin = config.spin
    mol.verbose = config.verbose
    mol.build()

    # RHF
    rhf_obj = scf.RHF(mol).run()
    rhf_energy = float(rhf_obj.e_tot)

    # UHF with stability check
    uhf_obj = scf.UHF(mol).run()
    mo1 = uhf_obj.stability()[0]
    dm1 = uhf_obj.make_rdm1(mo1, uhf_obj.mo_occ)
    uhf_obj = uhf_obj.run(dm1)
    uhf_energy = float(uhf_obj.e_tot)

    # FCI using RHF reference
    ci_solver = fci.FCI(rhf_obj).run()
    fci_energy = float(ci_solver.e_tot)

    return HChainResult(
        fci_energy=fci_energy,
        rhf_energy=rhf_energy,
        uhf_energy=uhf_energy,
        geometry=config.geometry,
    )
