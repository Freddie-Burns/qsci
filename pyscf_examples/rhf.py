"""
Example of how to calculate RHF solutions with PySCF.
"""

from pyscf import gto, scf

# --- 1) Build molecule ---
# H2 at 0.74 Å (typical bond length). Change distance if you want.
mol = gto.Mole()
mol.atom = "H 0 0 0; H 0 0 0.74"    # in Angstrom by default
mol.basis = "STO-3G"
mol.charge = 0
mol.spin = 0   # number of unpaired electrons: 0 => singlet (default)
mol.verbose = 4
mol.build()

# --- 2) RHF calculation ---
mf_rhf = scf.RHF(mol)
e_rhf = mf_rhf.kernel()   # returns total SCF energy
print("\nRHF total energy (E):", e_rhf)
print("RHF orbital energies (e):", mf_rhf.mo_energy)
print("RHF occupation numbers:", mf_rhf.mo_occ)

# --- 3) UHF calculation ---
# For neutral H2 singlet we keep mol.spin=0; UHF will be spin-unrestricted
mf_uhf = scf.UHF(mol)
e_uhf = mf_uhf.kernel()
print("\nUHF total energy (E):", e_uhf)
print("UHF alpha orbital energies:", mf_uhf.mo_energy[0])
print("UHF beta  orbital energies:", mf_uhf.mo_energy[1])
print("UHF alpha occupations:", mf_uhf.mo_occ[0])
print("UHF beta  occupations:", mf_uhf.mo_occ[1])
