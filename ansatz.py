"""
Create ansatz circuit for sampling on the quantum device.
"""


import ffsim
from qiskit import QuantumCircuit, QuantumRegister, transpile
from pyscf.cc import CCSD


def get_lucj_circuit(
        ccsd_obj: CCSD,
        backend,
        n_reps: int = 1, 
        pairs_aa: list[tuple[int]] = None, 
        pairs_ab: list[tuple[int]] = None,
        ab_coupling_interval: int = 1, # only used if pairs_ab not set
        homo_lumo_expansion: bool = True
    ) -> QuantumCircuit:
    """
    Create a local unitary cluster Jastrow ansatz circuit.
    """
    # Number of orbitals and electrons in the molecule object
    norb, nelec = int(ccsd_obj.mol.nao), ccsd_obj.mol.nelec

    # Initialize from ccsd t2 amplitudes (calculated with PySCF)
    ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(ccsd_obj.t2, n_reps=n_reps)
    if pairs_aa is None:
        # Setting up square "ladder" interaction graph, same-spin couplings on a line topology:
        pairs_aa = [(p, p + 1) for p in range(norb - 1)]
    if pairs_ab is None:
        # Diagonal Coulomb interactions between up/down electrons sharing same spatial orbital
        pairs_ab = [(p,p) for p in range(0,norb,ab_coupling_interval)]
    interaction_pairs = (pairs_aa, pairs_ab)

    # Enforce locality constraints (i.e. to respect the interaction graph specified above):
    lparams = ucj_op.to_parameters(interaction_pairs=interaction_pairs)
    lucj_op = ffsim.UCJOpSpinBalanced.from_parameters(
        lparams,
        norb=norb,
        n_reps=n_reps,
        interaction_pairs=interaction_pairs,
        with_final_orbital_rotation=False
    )

    # Create a QuantumCircuit, initialized from the HF state:
    qubits = QuantumRegister(2 * norb, name="q")
    circuit = QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)#; circuit.barrier()
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(lucj_op), qubits)
    circuit = circuit.decompose().decompose()

    if homo_lumo_expansion:
        # lightcone optimization
        n_gates = len(circuit.data)
        n_gates_prior = None
        while n_gates != n_gates_prior:
            n_gates_prior = n_gates
            drop_gate_indices = []
            for index,gate in enumerate(circuit.data):
                if gate.name == "xx_plus_yy":    
                    gate_qubits = {q._index for q in gate.qubits}
                    gate_intersections = [gate_prior.name for gate_prior in circuit.data[:index] if gate_prior.name != 'barrier' and gate_qubits.intersection({q._index for q in gate_prior.qubits})!=set()]
                    if gate_intersections == [] or gate_intersections == ['x', 'x']:
                        drop_gate_indices.append(index)
            drop_gate_indices = list(sorted(set(drop_gate_indices)))
            for i,index in enumerate(drop_gate_indices):
                circuit.data.pop(index-i)
            n_gates = len(circuit.data)

    # reorder spin convention uuuu...dddd... -> udududud...
    qc = transpile(circuit, initial_layout={q:2*(i%norb)+(i//norb) for i,q in enumerate(qubits)}, optimization_level=0)
    qc = qc.copy() # Not sure why, don't question it I guess
    qc.measure_all()
    return transpile(qc, backend=backend, optimization_level=3)
