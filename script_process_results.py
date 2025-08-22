"""
Download bitstrings, calculate energies, save results.
"""


import energy
import store


job_data: store.JobData = store.load_job_data("e2579a")
fci_energies = []
qsci_energies = []

for molecule, result in zip(job_data.molecules, job_data.results):
    fci, qsci = energy.calculate_energy_ladders(molecule, result)
    fci_energies.append(fci)
    qsci_energies.append(qsci)
