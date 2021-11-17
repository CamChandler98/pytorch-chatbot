import torch
import random
from processed_data import voc, pairs
from word_processing import eos_token, pad_token
from dependencies import itertools

def indexes_from_sentence(voc, sentence):
    return [voc.word_to_index[word] for word in sentence.split(' ') ] + [eos_token]


inp = []
out = []

for pair in pairs [:10]:
    inp.append(pair[0])
    out.append(pair[1])


indexes = [indexes_from_sentence(voc,sentence) for sentence in inp]



def zero_padding(lst , fill_value = 0):
    return list(itertools.zip_longest(*lst, fillvalue=fill_value))


def binary_matrix(lst, value = 0):
    matrix = []
    for i, seq in enumerate(lst):
        matrix.append([])
        for token in seq:
            if token == pad_token:
                matrix[i].append(0)
            else:
                matrix[i].append(1)

    return matrix

test = zero_padding(indexes)

binary_mat = binary_matrix(test)



def input_var(lst, voc):
    indexes_batch = [indexes_from_sentence(voc, sentence) for sentence in lst]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padded_lst = zero_padding(indexes_batch)
    pad_var = torch.LongTensor(padded_lst)

    return pad_var, lengths

def output_var(lst, voc):
    indexes_batch = [indexes_from_sentence(voc, sentence) for sentence in lst]
    max_target_length = max([len(indexes) for indexes in indexes_batch])
    padded_lst = zero_padding(indexes_batch)
    mask = binary_matrix(padded_lst)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padded_lst)
    return padVar, mask, max_target_length


def batch_to_train_data(voc, pair_batch):
    pair_batch.sort(key = lambda sen: len(sen[0].split(' ')), reverse= True)
    input_batch, output_batch = [], []

    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])

    inp, lengths = input_var(input_batch, voc)

    output , mask, max_target_length = output_var(output_batch, voc)

    return inp, lengths, output, mask, max_target_length



# batch_size = 5

# batches = batch_to_train_data(voc, [random.choice(pairs) for _ in range(batch_size)])

# input_variable , lengths, ovar, mask , max_len = batches

# print('i')
# print(input_variable)
# print(' ')
# print('lengths')
# print(lengths)
# print('targets')
# print(ovar)
# print('mask')
# print (mask)
# print('max length')
# print (max_len)
