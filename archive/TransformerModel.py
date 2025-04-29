from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, LayerNormalization, Dropout
from tensorflow.keras.models import Model


def build_transformer_model(input_shape):
    inputs = Input(shape=input_shape)
    x = MultiHeadAttention(num_heads=4, key_dim=50)(inputs, inputs)
    x = LayerNormalization(epsilon=1e-6)(x + inputs)
    x = Dense(100, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation='relu')(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model


# Train Transformer
transformer_model = build_transformer_model((look_back, X.shape[2]))
transformer_model.fit(X, y, epochs=10, batch_size=32)
transformer_predictions = transformer_model.predict(X)
transformer_predictions = scaler.inverse_transform(
    np.concatenate((transformer_predictions, np.zeros((len(transformer_predictions), X.shape[2] - 1))), axis=1))[:, 0]
transformer_trades, transformer_cash = trading_strategy(transformer_predictions, data)
print("Transformer Trades:", transformer_trades)
print("Transformer Final Cash:", transformer_cash)
