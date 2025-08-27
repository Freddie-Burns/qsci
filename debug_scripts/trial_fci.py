import numpy as np
import matplotlib.pyplot as plt

import energy
import molecule

trihydrogen = molecule.trihydrogen_cation(1, basis="sto-3g")
energy.fci_energy(trihydrogen)