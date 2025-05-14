"""
Top level script to run the quantum circuit on the hardware.
"""


import numpy as np

from ansatz import get_lucj_circuit, transpile
from cloud import CloudURL, get_backend
from molecule import create_molecule, coupled_cluster

import store

# Parameters of the run
# Shots, the number of times to run circuit and measure
# URL to the chosen IQM quantum hardware
N_SHOTS = 2**13
CLOUD_URL = CloudURL.garnet


# Define backend
backend = get_backend(cloud_url=CLOUD_URL)

# Build molecules
molecules = []
ccsds = []
for i in np.arange(0.2, 2.2, 0.2):
    geometry = [('Li',(0,0,0)),('H',(0,0,i))]
    print(geometry)
    molecule = create_molecule(geometry=geometry)
    ccsd = coupled_cluster(molecule, run_fci=True)
    molecules.append(molecule)
    ccsds.append(ccsd)

# Create circuits
# circuits = []
# for ccsd in ccsds:
#     circuits.append(get_lucj_circuit(ccsd_obj=ccsd, backend=backend))

# Run the job online
# job = backend.run(circuits, shots=N_SHOTS)

# Store the parameters of the job using JobData class
# This saves the data to a .pkl file on instantiation
job_data = store.JobData(
    molecules=molecules,
    job_id="0196aac1-3398-7c12-bcd4-4a9aaf8ef116",
    cloud_url=CLOUD_URL,
    n_shots=N_SHOTS,
)

# Save job parameters
# These are required for the data processing once the job is completed
