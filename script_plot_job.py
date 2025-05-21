"""
Download the finished job and process it.
"""


from pathlib import Path

import dill
import numpy as np
from matplotlib import pyplot as plt

import store


job_data: store.JobData = store.load_job_data("782d95")
data_path: Path = store.job_data_path(job_data.job_id)

bond_lengths = []
for mol in job_data.molecules:
    bond_lengths.append((round(mol.geometry[1][1][2], 1)))

with open(data_path / "qsci_energies.pkl", 'rb') as file:
    qsci_energies = dill.load(file)
with open(data_path / "fci_energies.pkl", 'rb') as file:
    fci_energies = dill.load(file)

qsci_pes = qsci_energies - np.min(qsci_energies)
fci_pes = fci_energies - np.min(fci_energies)

fig = plt.figure()

for i, state in enumerate(qsci_pes):
    plt.plot(bond_lengths, state, label=str(i))
plt.legend()

# Reset colour cycle so FCI and QSCI colours correspond
plt.gca().set_prop_cycle(None)
for i, state in enumerate(fci_pes):
    plt.plot(bond_lengths, state, ':', label=str(i))

plt.xlabel('Bond Length (Angstrom)')
plt.ylabel('Energy (eV)')
plt.title('LiH Bond Stretch QSCI')
plt.xlim(0, 6)
plt.show()
