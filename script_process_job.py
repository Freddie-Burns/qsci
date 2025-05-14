"""
Download the finished job and process it.
"""

import dill

import matplotlib.pyplot as plt
import numpy as np

import energy

with open("data/JobSaver-0196aac1-3398-7c12-bcd4-4a9aaf8ef116.pkl", 'rb') as file:
    job_data = dill.load(file)

bond_lengths = []
for mol in job_data.molecules:
    bond_lengths.append((round(mol.geometry[1][1][2], 1)))



fci_energies = []
qsci_energies = []
for molecule, result in zip(job_data.molecules, job_data.results):
    fci, qsci = energy.calculate_energy_ladders(molecule, result)
    fci_energies.append(fci)
    qsci_energies.append(qsci)

qsci_stretch = np.array(list(zip(*qsci_energies)))
fci_stretch = np.array(list(zip(*fci_energies)))
qsci_error = qsci_stretch - fci_stretch

with open("qsci_error.pkl", 'wb') as file:
    dill.dump(qsci_error, file)

with open("qsci_error.pkl", 'rb') as file:
    qsci_error = dill.load(file)

bond_lengths = []
for mol in job_data.molecules:
    bond_lengths.append((round(mol.geometry[1][1][2], 1)))

fig1 = plt.figure()
for i, state in enumerate(qsci_error):
    plt.plot(bond_lengths, state, label=str(i))
plt.axhline(0.043, color='black')
plt.legend()
plt.xlabel('Bond Length (Angstrom)')
plt.ylabel('Error (eV)')
plt.title('LiH Bond Stretch QSCI')
fig1.savefig("qsci_error.png")

fig2 = plt.figure()
for i, state in enumerate(qsci_error[:-3]):
    plt.plot(bond_lengths, state, label=str(i))
plt.axhline(0.043, color='black')
plt.legend()
plt.xlabel('Bond Length (Angstrom)')
plt.ylabel('Error (eV)')
plt.title('LiH Bond Stretch QSCI')
plt.show()