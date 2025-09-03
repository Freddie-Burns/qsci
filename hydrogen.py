"""
Examples
--------
>>> cfg = HChainConfig(n_atoms=6, bond_length=0.75)
>>> res = calculate_h_chain_energies(cfg)
>>> res.fci_energy
-3.14
"""


import numpy as np
from pydantic import BaseModel, Field, computed_field
from pyscf import gto, scf, fci

from molecule import Basis


class HChainConfig(BaseModel):
    """
    Configuration model for building and computing a linear hydrogen chain.

    This class validates input parameters such as the number of atoms and
    bond length, and provides derived quantities like the spin multiplicity.

    Parameters
    ----------
    n_atoms : int
        Number of hydrogen atoms in the chain. Must be >= 1.
    bond_length : float
        Distance between neighbouring hydrogen atoms in angstroms. Must be > 0.
    basis : Basis, default=Basis.STO_3G
        Basis set to use for the calculation.
    charge : int, default=0
        Net charge of the molecule.
    verbose : int, default=1
        PySCF verbosity level (0 = silent, higher values = more output).

    Attributes
    ----------
    spin : int
        Number of unpaired electrons. Computed as `n_atoms % 2`.
    """
    n_atoms: int = Field(..., ge=1, description="Number of H atoms")
    bond_length: float = Field(..., gt=0, description="Å")
    basis: Basis = Field(default=Basis.STO_3G)
    charge: int = 0
    verbose: int = 1

    @property
    def spin(self) -> int:
        """Number of unpaired electrons; 0 indicates a singlet state."""
        return self.n_atoms % 2


class HChainResult(BaseModel):
    """
    Result model for electronic structure calculations on a hydrogen chain.

    Stores the total electronic energies computed by FCI, RHF, and UHF methods,
    along with the final molecular geometry specification.

    Parameters
    ----------
    fci_energy : float
        Total energy from the full configuration interaction (FCI) calculation,
        in Hartree.
    rhf_energy : float
        Total energy from the restricted Hartree–Fock (RHF) calculation,
        in Hartree.
    uhf_energy : float
        Total energy from the unrestricted Hartree–Fock (UHF) calculation,
        in Hartree.
    geometry : str
        PySCF-compatible geometry string of the hydrogen chain, formatted as
        `"H 0 0 z; H 0 0 z; ..."`.
    z_positions: list[float]
        z-axis positions of the molecule, in angstroms.
    bond_length: float
        Bond length of first two hydrogens in the chain in angstroms.
    """
    fci_energy: float
    """FCI total energy in Hartree."""

    rhf_energy: float
    """RHF total energy in Hartree."""

    uhf_energy: float
    """UHF total energy in Hartree."""

    geometry: str
    """Geometry string of the hydrogen chain in PySCF format."""

    # Standard properties are not stored in a pandas.DataFrame by model_dump
    # A computed_field will be stored.
    @computed_field(return_type=list[float])
    def z_positions(self) -> list[float]:
        """Array of z-axis positions of hydrogen atoms in the chain."""
        return [float(atom.split()[3]) for atom in self.geometry.split(";")]

    @computed_field(return_type=float)
    def bond_length(self) -> float:
        """Bond length of first two hydrogen atoms in the chain."""
        return self.z_positions[1] - self.z_positions[0]


def calculate_h_chain_energies(config: HChainConfig) -> HChainResult:
    """
    Build and compute the electronic structure of a linear hydrogen chain.

    Constructs a chain of hydrogen atoms aligned along the z-axis with equal
    spacing, using the parameters provided in `HChainConfig`. The function
    runs restricted Hartree–Fock (RHF), unrestricted Hartree–Fock (UHF), and
    full configuration interaction (FCI) calculations using PySCF, returning
    their total energies.

    Parameters
    ----------
    config : HChainConfig
        Configuration object specifying the number of atoms, bond length,
        basis set, and other molecular parameters.

    Returns
    -------
    HChainResult
        Object containing the RHF, UHF, and FCI total energies in Hartree,
        as well as the geometry string.

    Notes
    -----
    - The molecule is assumed to be neutral (`charge=0`) unless overridden.
    - The spin multiplicity is set automatically from `n_atoms % 2`.
    - Energies are returned in atomic units (Hartree).

    Examples
    --------
    >>> cfg = HChainConfig(n_atoms=6, bond_length=0.75)
    >>> res = calculate_h_chain_energies(cfg)
    >>> res.fci_energy
    -3.14159
    """
    geometry = "; ".join(
        f"H 0 0 {n * config.bond_length:.6f}" for n in range(config.n_atoms)
    )

    mol = gto.Mole()
    mol.atom = geometry
    mol.basis = config.basis
    mol.charge = config.charge
    mol.spin = config.spin
    mol.verbose = config.verbose
    mol.build()

    rhf_obj = scf.RHF(mol).run()
    rhf_energy = rhf_obj.e_tot

    uhf_obj = scf.UHF(mol).run()
    mo1 = uhf_obj.stability()[0]
    dm1 = uhf_obj.make_rdm1(mo1, uhf_obj.mo_occ)
    uhf_obj = uhf_obj.run(dm1)
    uhf_energy = uhf_obj.e_tot

    ci_solver = fci.FCI(rhf_obj).run()
    fci_energy = ci_solver.e_tot

    return HChainResult(
        fci_energy=fci_energy,
        rhf_energy=rhf_energy,
        uhf_energy=uhf_energy,
        geometry=geometry,
    )
