"""
Module to assist interfacing with the quantum hardware cloud services
such as those provided by IQM Resonance.
"""


from enum import Enum

from iqm.iqm_client import IQMClient
from iqm.qiskit_iqm import IQMBackend


with open("token.txt", 'r') as textfile:
    TOKEN = textfile.read()


class CloudURL(Enum):
    """
    Enumerator for the URLs of the quantum devices.
    """
    emerald = "https://cocos.resonance.meetiqm.com/emerald-fe33b793-44a0-42d6"
    garnet = "https://cocos.resonance.meetiqm.com/garnet"
    sirius = "https://cocos.resonance.meetiqm.com/sirius"

    def __get__(self, *args):
        """
        Return the url string when accessed by attribute.
        """
        return self.value


def get_backend(cloud_url: CloudURL) -> IQMBackend:
    """
    Return IQM backend object for the given cloud URL.
    """
    client = IQMClient(cloud_url, token=TOKEN)
    return IQMBackend(client=client)
