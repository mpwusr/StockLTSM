import pennylane as qml
import pennylane.numpy as qnp
import numpy as np

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
    for i in range(n_qubits):
        qml.RY(weights[i], wires=i)
    return qml.expval(qml.PauliZ(0))

def optimize_quantum_weights(X, y, iterations=10):
    weights = qnp.random.random(n_qubits, requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=0.1)
    y = qnp.array(y, requires_grad=False)
    for _ in range(iterations):
        def cost(w):
            preds = [quantum_circuit(x[:n_qubits], w) for x in X[:, -1]]
            return qnp.mean((qnp.array(preds) - y) ** 2)
        weights = opt.step(cost, weights)
    return weights

def quantum_predict_future(last_sequence, weights, scaler, look_back, future_days):
    predictions = []
    current_sequence = last_sequence.copy()
    for _ in range(future_days):
        pred = quantum_circuit(current_sequence[-1, :n_qubits], weights)
        predictions.append(pred)
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1, 0] = pred
    predictions = (np.array(predictions) + 1) / 2
    padded = np.concatenate((predictions.reshape(-1, 1), np.zeros((len(predictions), 6))), axis=1)
    return np.maximum(scaler.inverse_transform(padded)[:, 0], 0)
