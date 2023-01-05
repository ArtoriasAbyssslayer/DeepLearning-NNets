    
# TEST MODEL
#     
# Use the encoder model to encode the input data
encoder = autoencoder
encoded_inputs = encoder.forward(x_test)

# Use the decoder model to decode the encoded data
decoded_outputs = autoencoder.forward(encoded_inputs)

# Print the original and decoded data
print(x_test[0])
print(decoded_outputs[0])

# TODO 







