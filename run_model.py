import torch
import os
from training import voc, optim, pairs, trainIters
from training import NeuralNetwork, EncoderRNN, LuongAttnDecoderRNN, device
# Configure models
model_name = 'cb_model'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64


checkpoint = None
encoder_sd = None
decoder_sd = None
encoder_optimizer_sd = None
decoder_optimizer_sd = None
embedding_sd = None

loadFilename = True
checkpoint_iter = 1000

if loadFilename:
    print('loading checkpoint ')
    checkpoint = torch.load('./data/save/cb_model/movie_data/2-2_500/2500_checkpoint.tar')
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']

print('Building encoder and decoder ...')

embedding = NeuralNetwork.Embedding(voc.num_words, hidden_size)

if loadFilename:
    embedding.load_state_dict(embedding_sd)


encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')


corpus_name = 'movie_data'
save_dir = os.path.join("data", "save")
clip = 50.0
teacher_forcing_ratio = .5
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 5000
print_every = 10
save_every = 500

print('set training mode')
encoder.train()
decoder.train()
print('in training mode')

print("optimizing rnn's")
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
print("rnn optimized")
if loadFilename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)


trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
           embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
           print_every, save_every, clip, corpus_name,  loadFilename, checkpoint = checkpoint)
