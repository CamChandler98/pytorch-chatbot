import torch
from word_processing import sos_token
from loss import maskNLLLoss
from data_preparation import batch_to_train_data, indexes_from_sentence
from processed_data import voc, pairs
from dependencies import random, NeuralNetwork, optim, os
from encoder_rnn import EncoderRNN, LuongAttnDecoderRNN
from conversation_data_processing import device
from normalize_words import normalize_string, max_length as max_target_length

checkpoint = None

def train(input_variable, lengths, target_variable, mask, max_target_length, encoder, decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip, max_length = max_target_length, teacher_forcing_ratio = 1.0):

    #zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    #set device options
    input_variable = input_variable.to(device)
    target_variable = target_variable.to(device)
    lengths = lengths.to('cpu')
    mask = mask.to(device)

    #initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    #forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths )


    # Create initial decoder input( start with sos tokens for each sentence)
    decoder_input = torch.LongTensor([[sos_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

        # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_length):
            # print('training')
            # print('decoder_input', decoder_input.size())
            # print('decoder_hidden',decoder_hidden.size())
            # print("encoder_outputs",encoder_outputs.size())
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)

            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1,-1)

            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])

            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    else:
        for t in range(max_target_length):
            decoder_output, decoder_hidden = decoder( decoder_input, decoder_hidden, encoder_outputs)

            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)

            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])

            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    loss.backward()

    _ = NeuralNetwork.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = NeuralNetwork.utils.clip_grad_norm_(decoder.parameters(), clip)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


def trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size, print_every, save_every, clip, corpus_name, loadFilename, checkpoint = None):

    training_batches =  [batch_to_train_data(voc, [random.choice(pairs) for _ in range(batch_size)]) for _ in range(n_iteration)]

    print('Initializing')

    start_iteration = 1
    print_loss = 0

    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1

    print('training...')

    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]

        #extract fields from batch
        input_variable , lengths, target_variable, mask , max_target_length = training_batch

        loss = train(input_variable, lengths, target_variable, mask, max_target_length, encoder, decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)

        print_loss += loss

        if iteration % print_every ==0:

            print_loss_avg = print_loss / iteration

            print(f'Iteration: {iteration}')
            print(f'percent complete: {(iteration/n_iteration) * 100 }')
            print(f'average loss {print_loss_avg}')

                # Save checkpoint
        if (iteration % save_every == 0):
            "saving"
            directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, '500'))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))


class GreedySearchDecoder(NeuralNetwork.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * sos_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores


def evaluate(encoder, decoder, searcher, voc, sentence, max_length=max_target_length):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexes_from_sentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to("cpu")
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index_to_word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = normalize_string(input_sentence)
            # Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")
