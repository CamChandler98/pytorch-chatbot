from encoder_rnn import EncoderRNN, LuongAttnDecoderRNN, NeuralNetwork
from processed_data import voc
from dependencies import optim
from training import GreedySearchDecoder, evaluateInput
import torch
import os

print('initializing')
attn_model = 'dot'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64
learning_rate = 0.0001
decoder_learning_ratio = 5.0


embedding = NeuralNetwork.Embedding(voc.num_words, hidden_size)
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)

print('optimizing')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)


bot = torch.load("trained_bot/bot.tar")


embedding.load_state_dict(bot['embedding'].state_dict())
encoder.load_state_dict(bot['encoder'].state_dict())
decoder.load_state_dict(bot['decoder'].state_dict())
encoder_optimizer.load_state_dict(bot['en_opt'].state_dict())
decoder_optimizer.load_state_dict(bot['de_opt'].state_dict())
torch.save({
    "embedding": embedding.state_dict(),
    "encoder" : encoder.state_dict(),
    "decoder": decoder.state_dict(),
    "en_opt": encoder_optimizer.state_dict(),
    "de_opt": decoder_optimizer.state_dict(),
}, os.path.join('trained_bot', 'bot.tar'))
