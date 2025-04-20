import pennylane as qml
import pennylane.numpy as qnp
import numpy as np

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    # Encode inputs
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
        qml.RZ(inputs[i], wires=i)

    # Variational layers
    for layer in range(4):
        for i in range(n_qubits):
            qml.RX(weights[layer * n_qubits + i], wires=i)
            qml.RZ(weights[layer * n_qubits + i], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])

    return qml.expval(qml.PauliZ(0))

def normalize_input(x):
    # Handle both 1D (single time step) and 2D (sequence) inputs
    if len(x.shape) == 1:
        raw = x[:n_qubits]
    else:
        raw = np.mean(x, axis=0)[:n_qubits]  # Average over time steps

    min_val = np.min(raw)
    max_val = np.max(raw)
    scaled = (raw - min_val) / (max_val - min_val + 1e-6)
    print(f"Normalized input (first {n_qubits}):", scaled)
    return (scaled * 2 * np.pi) - np.pi

def smooth_predictions(preds, window=3):
    return np.convolve(preds, np.ones(window) / window, mode="same")

def optimize_quantum_weights(X, y, iterations=300, verbose=False):
    weights = qnp.random.random(4 * n_qubits, requires_grad=True)  # Adjusted for 4 layers
    opt = qml.AdamOptimizer(stepsize=0.1)  # Reverted to AdamOptimizer
    y = qnp.array(y, requires_grad=False)
    for step in range(iterations):
        def cost(w):
            preds = []
            for x in X[:, -1]:
                x_scaled = normalize_input(x)
                pred = quantum_circuit(x_scaled, w)
                preds.append(pred)
            preds = qnp.array(preds)
            if verbose and step % 10 == 0:
                print(f"[Step {step}] Sample preds:", preds[:5])
            return qnp.mean((preds - y) ** 2)
        weights = opt.step(cost, weights)
    return weights

def quantum_predict_future(last_sequence, weights, scaler, look_back, future_days, horizon="Short"):
    predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(future_days):
        x_scaled = normalize_input(current_sequence)
        pred = quantum_circuit(x_scaled, weights)
        predictions.append(pred)
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1, 0] = pred

    predictions = np.array(predictions)
    print(f"[Horizon: {horizon}] Raw predictions (first 10):", predictions[:10])

    predictions = (predictions + 1) / 2  # Normalize to [0, 1]
    print(f"[Horizon: {horizon}] Normalized predictions (first 10):", predictions[:10])

    # Removed smooth_predictions to allow more variability
    print(f"[Horizon: {horizon}] Predictions after normalization (first 10):", predictions[:10])

    padded = np.concatenate((predictions.reshape(-1, 1), np.zeros((len(predictions), 6))), axis=1)
    final_predictions = np.maximum(scaler.inverse_transform(padded)[:, 0], 0)
    print(f"[Horizon: {horizon}] Final predictions (first 10):", final_predictions[:10])

    return final_predictions