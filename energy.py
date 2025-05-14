"""
Module for energy calculation functionality.
Configuration selection and diagonalisation of the Hamiltonian.
"""


from collections import Counter
from math import comb
from typing import List

import numpy as np
from iqm.qiskit_iqm import IQMJob
from openfermion import MolecularData, jordan_wigner
from openfermionpyscf import run_pyscf
from pyscf.fci import FCI
from pyscf.gto import Mole
from pyscf.scf import RHF, UHF
from qiskit.result.models import ExperimentResult
from symmer import PauliwordOp, QuantumState

from store import JobData


EV_RANGE = 10
EV_FACTOR = 27.2114


def calculate_energy_ladders(
        molecule: MolecularData,
        result: ExperimentResult,
    ) -> tuple:
    """
    Params:
    molecular_index: int, the index of the molecule in self.molecules list.
    Returns:
    tuple: fci_ladder, qsci_ladder

    Calculate the QSCI and FCI energy ladders.
    Returns energy ladders for FCI and QSCI calculations in eV.
    """
    qsci_ladder = qsci_energies(molecule, result)
    fci_ladder = fci_energy(molecule)

    # Convert Hartree to eV
    fci_ladder *= EV_FACTOR
    # Set FCI ground state as 0 eV
    ground = fci_ladder[0]
    fci_ladder -= ground
    # Only look at states within EV_RANGE of ground
    lowest_fci_mask = fci_ladder < EV_RANGE
    fci_ladder = fci_ladder[lowest_fci_mask]

    # Repeat for QSCI energies
    qsci_ladder *= EV_FACTOR
    qsci_ladder -= ground
    lowest_eigvals_mask = qsci_ladder < EV_RANGE
    lowest_eigvals = qsci_ladder[lowest_eigvals_mask]
    qsci_ladder = lowest_eigvals

    return fci_ladder, qsci_ladder


def qsci_energies(
        molecule: MolecularData,
        result: ExperimentResult,
    ) -> tuple[float]:
    """
    Params:
    molecule_index: int, index of the molecule in self.molecules list.
    Returns:
    list, the energy eigenvalues in units of Hartree.

    Use QSCI to calculate the energy eigenvalues of the Hamiltonian for a
    molecule. Select the most important configurations from the job's
    measurement results then diagonalise the Hamiltonian in this subspace.
    """
    # Pauli Hamiltonian for this molecule and geometry
    H = calculate_hamiltonian(molecule)
    # Find the most frequent configurations from the mmnt data
    configurations = _select_configurations(molecule, result)
    # Reduce the Hamiltonian matrix to the subspace of these configs
    selected_subspace_matrix = get_selected_configuration_matrix(H, configurations)
    # Diagonalise this matrix to find the energy eigen values
    eigvals, eigvecs = np.linalg.eigh(selected_subspace_matrix)
    return eigvals


def fci_energy(
        molecule: MolecularData,
    ) -> list[float]:
    """
    :param molecule: MolecularData for the molecule being simulated.
    :return: list[float], FCI energies in Hartree.
    """
    ensure_pyscf_calculated(molecule)
    mol: Mole = molecule._pyscf_data['mol']
    hf: RHF | UHF = molecule._pyscf_data['scf']
    fci_energies = calculate_fci(mol, hf)
    return fci_energies


def ensure_pyscf_calculated(molecule: MolecularData) -> None:
    """
    Check for a _pyscf_data attribute and if not present, run pyscf.
    This is done in place so no need to return anything.
    """
    if vars(molecule).get('_pyscf_data') is None:
        run_pyscf(molecule)

def calculate_fci(mol, hf):
    """
    Calculate the full configuration interaction energies as a basis for comparison.
    """
    fci = FCI(hf)
    total_configs = comb(mol.nao, mol.nelec[0]) * comb(mol.nao, mol.nelec[1])
    fci.nroots = total_configs # maximum number of configurations
    fci_energies, _ = fci.kernel()
    return fci_energies


def calculate_hamiltonian(molecule: MolecularData):
    """
    Create Hamiltonian of Pauli operators from molecular data.
    """
    H = jordan_wigner(molecule.get_molecular_hamiltonian())
    return PauliwordOp.from_openfermion(H, n_qubits=molecule.n_orbitals*2)


def get_selected_configuration_matrix(
        hamiltonian: PauliwordOp,
        configs: List[str]
    ) -> np.ndarray[float]:
    """
    Constructs the interaction matrix H_ij = <i|H|j>
    """
    matrix = np.zeros((len(configs),len(configs)),dtype=float)
    # this matrix is symmetric, so only build the upper triangle and then mirror
    for i,i_str in enumerate(configs):
        for j,j_str in enumerate(configs[i:]):
            matrix[i,j+i] = matrix_element(hamiltonian, i_str, j_str)
    matrix[np.tril_indices(len(configs), -1)] = matrix.T[np.tril_indices(len(configs), -1)]
    return matrix


def matrix_element(hamiltonian: PauliwordOp, i_str: str, j_str: str) -> float:
    psi_i = QuantumState.from_dictionary({i_str:1})
    psi_j = QuantumState.from_dictionary({j_str:1})
    return (psi_i.dagger * hamiltonian * psi_j).real


def _select_configurations(
        molecule: MolecularData,
        result: ExperimentResult,
        max_configs: int=200,
    ) -> List[str]:
    """
    Given a job object, select the most important configurations from the results.
    """
    measurement_dict = result.data.counts
    # if the measurements are in hex, convert to bin: (this is the case when using a FakeBackend)
    if any([not set(m).issubset({'0','1'}) for m in measurement_dict]):
        measurement_dict = {format(int(k, base=16), f'0{molecule.n_electrons * 2}b'):v for k,v in measurement_dict.items()}

    # reverse order of qubits:
    measurement_dict = {k[::-1]:v for k,v in measurement_dict.items()}

    # postselect configs with correct symmetry:
    ensure_pyscf_calculated(molecule)
    pyscf_mol: Mole = molecule._pyscf_data['mol']
    measurement_dict = {k:v for k,v in measurement_dict.items() if valid_config(k, pyscf_mol)}

    # reorder in decreasing measurement frequency
    measurement_dict = dict(sorted(measurement_dict.items(), key=lambda x:x[1])[::-1])
    configurations, frequencies = zip(*measurement_dict.items())
    configurations = list(configurations[:max_configs]) # truncate the configuration space, noting these are ordered by measurement frequency

    # manually append the Hartree-Fock config if missing:
    hf_str = '1' * molecule.n_electrons + '0' * (2 * molecule.n_orbitals - molecule.n_electrons)
    if not hf_str in configurations:
        configurations.append(hf_str)

    print(
        f"\nPercentage of measurements in the correct particle sector: "
        f"{100 * sum(frequencies) / result.shots: .3f} %"
    )

    return configurations


def valid_config(config: str, mol: Mole) -> bool:
    """
    Validate the correct number of alpha/beta particles.
    PySCF Mole object required for the nelec variable, a tuple of (alpha, beta).
    """
    alpha_str = config[::2]
    beta_str = config[1::2]
    return (Counter(alpha_str)['1'], Counter(beta_str)['1']) == mol.nelec
