from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from pyscf.cc import CCSD


def create_molecule(
        geometry,
        basis="sto-3g",
        multiplicity=1,
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
