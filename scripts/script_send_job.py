"""
Top level script to run the quantum circuit on the hardware.
"""


import cloud
import molecule
import store
from ansatz import get_lucj_circuit


# Parameters of the run
# Shots, the number of times to run circuit and measure
# URL to the chosen IQM quantum hardware
N_SHOTS = 2**14


# Define backend
cloud_url = cloud.CloudURL.emerald
backend = cloud.get_backend(cloud_url=cloud_url)

# Build molecules
molecules = []
ccsds = []
for i in [1.4]:
    i = round(i, 1)
    formaldehyde = molecule.formaldehyde(bond_length=i)
    ccsd = molecule.coupled_cluster(formaldehyde, run_fci=False)
    molecules.append(formaldehyde)
    ccsds.append(ccsd)

# Create circuits
circuits = []
for ccsd in ccsds:
    circuits.append(get_lucj_circuit(ccsd_obj=ccsd, backend=backend))

# Run the job online
job = backend.run(circuits, shots=N_SHOTS)

# Store the parameters of the job using JobData class
# This saves the data to a .pkl file on instantiation
job_data = store.JobData(job=job, molecules=molecules)

# # Retrieve job to resave job_data if edits are required
# job = backend.retrieve_job("0196ed52-5150-7442-89bd-ff49276d867e")
# job_data = store.JobData(job=job, molecules=molecules)

job_data.save_to_pkl()
