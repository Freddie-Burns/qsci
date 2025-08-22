import numpy as np
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from pyscf.cc import CCSD


def create_molecule(
        geometry,
        basis="sto-3g",
        multiplicity=0,
        charge=0,
    ) -> MolecularData:
    """
    geometry takes the form e.g. [('Li',(0,0,0)),('H',(0,0,1))]
    """
    return MolecularData(geometry, basis, multiplicity, charge)


def coupled_cluster(molecule, run_fci=False):
    molecule = run_pyscf(molecule, run_scf=True, run_fci=run_fci)
    hf = molecule._pyscf_data['scf']
    ccsd = CCSD(hf).run()
    return ccsd

def lithium_hydride(bond_length=1, **kwargs):
    geometry = [('Li',(0,0,0)),('H',(0,0,bond_length))]
    return create_molecule(geometry, multiplicity=1, charge=0, **kwargs)

def formaldehyde(bond_length=1, **kwargs):
    # Geometry from PubChem O C H H
    geometry_arr = np.array([
    [2.000, -0.560, 0.000],
    [2.866, -0.060, 0.000],
    [3.403, -0.370, 0.000],
    [2.866,  0.560, 0.000],
    ])

    # Move oxygen for desired C=O bond length
    bond_vector = geometry_arr[0] - geometry_arr[1]
    bond_vector /= np.linalg.norm(bond_vector)
    bond_vector *= bond_length
    geometry_arr[0] = geometry_arr[1] + bond_vector

    # Convert to list of tuples
    geometry_arr = [tuple(x) for x in geometry_arr]

    # Create PySCF molecule object
    geometry = list(zip("OCHH", geometry_arr))
    return create_molecule(geometry, charge=0, multiplicity=1, **kwargs)


def trihydrogen_cation(bond_length=1, **kwargs):
    """
    Three hydrogen atoms in an equilateral triangle on the z=0 plane.
    :param bond_length:
    :param kwargs:
    :return:
    """
    geometry = [
        ('H', (0, 0, 0)),
        ('H', (bond_length, 0, 0)),
        ('H', (bond_length / 2, bond_length * np.sqrt(3) / 2, 0))
    ]
    return create_molecule(geometry, charge=1, multiplicity=1, **kwargs)
