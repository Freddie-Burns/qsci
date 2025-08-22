"""
Save and load IQM job data.
Calculate energies by QSCI and plot the results.
"""


import os
from dataclasses import dataclass
from pathlib import Path

import dill # Extended pickle module that can deal with locals
from iqm.qiskit_iqm import IQMBackend
from iqm.qiskit_iqm.iqm_job import IQMJob
from openfermion.chem import MolecularData


ROOT = Path(__file__).parent.resolve()

# Save the API token in a text file
# Do not distribute or accidentally push to GitHub
with open(ROOT / "token.txt", 'r') as textfile:
    TOKEN = textfile.read()


@dataclass
class JobData:
    """
    IQMJob and openfermion MolecularData objects for saving to pickle.
    """
    job: IQMJob
    molecules: list[MolecularData]

    def save_to_pkl(self) -> None:
        """
        Save this JobData instance to a pkl file.
        """
        # Create data directory for saving to
        job_dir: Path = job_data_path(self.job_id)

        # Save this JobData instance as pkl file
        # Dill is extended pickle module that can deal with locals
        file_path: Path = job_dir / "JobData.pkl"
        with open(file_path, 'wb') as file:
            dill.dump(self, file)

    @property
    def backend(self) -> IQMBackend:
        return self.job.backend()

    @property
    def job_id(self) -> str:
        return self.job.job_id()

    @property
    def results(self):
        return self.job.result().results


def job_data_path(job_id: str) -> Path:
    """
    Create directory from final 6 characters of job ID.
    """
    job_dir: Path = ROOT / "data" / job_id[-6:]
    os.makedirs(job_dir, exist_ok=True)
    return job_dir


def load_job_data(job_id: str) -> JobData:
    """
    Load job data from pkl file.
    """
    job_dir: Path = job_data_path(job_id)
    with open(job_dir / "JobData.pkl", 'rb') as file:
        job_data = dill.load(file)
    return job_data
