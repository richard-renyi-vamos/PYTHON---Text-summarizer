import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model

# Define input sequence
encoder_inputs = Input(shape=(None, num_encoder_tokens))
# Define LSTM encoder
encoder = LSTM(latent_dim, return_state=True)
# Get encoder outputs and states
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# Discard encoder outputs, keep only states
encoder_states = [state_h, state_c]

# Set up the decoder, using encoder_states as initial state
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# Define LSTM decoder, return sequences
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
# Get decoder outputs and states
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
# Define dense layer to output summary
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# encoder_input_data & decoder_input_data into decoder_target_data
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
