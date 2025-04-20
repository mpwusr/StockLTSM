from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, GRU, Dense, Dropout, Input, MultiHeadAttention,
    LayerNormalization, Add, Conv1D, MaxPooling1D
)
from tensorflow.keras.metrics import MeanAbsoluteError, MeanAbsolutePercentageError


def build_lstm_model(input_shape):
    inputs = Input(shape=input_shape, name="input_layer")
    x = LSTM(50, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(50)(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=[
            MeanAbsoluteError(name='mae'),
            MeanAbsolutePercentageError(name='mape')
        ],
        run_eagerly=False
    )
    return model


def build_transformer_model(input_shape):
    inputs = Input(shape=input_shape, name="input_layer")
    attention_output = MultiHeadAttention(num_heads=4, key_dim=64)(inputs, inputs)
    attention_output = Add()([inputs, attention_output])
    x = LayerNormalization(epsilon=1e-6)(attention_output)

    ffn_output = Dense(128, activation="relu")(x)
    ffn_output = Dense(input_shape[-1])(ffn_output)
    x = Add()([x, ffn_output])
    x = LayerNormalization(epsilon=1e-6)(x)

    x = x[:, -1, :]  # extract last time step
    x = Dense(100, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation='relu')(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=[
            MeanAbsoluteError(name='mae'),
            MeanAbsolutePercentageError(name='mape')
        ],
        run_eagerly=False
    )
    return model


def build_gru_cnn_model(input_shape):
    inputs = Input(shape=input_shape, name="input_layer")
    x = Conv1D(filters=32, kernel_size=3, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = GRU(64, return_sequences=True)(x)
    x = GRU(32)(x)
    x = Dropout(0.2)(x)
    x = Dense(50, activation='relu')(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=[
            MeanAbsoluteError(name='mae'),
            MeanAbsolutePercentageError(name='mape')
        ],
        run_eagerly=False
    )
    return model
