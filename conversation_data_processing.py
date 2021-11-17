import torch
# import torch.nn as NeuralNetwork
# from torch import optim
# import torch.nn.functional as FunctionalNN
# import csv
# import random
# import re
# import os
# import unicodedata
# import codecs
# import itertools

cuda_is_available = torch.cuda.is_available()

device = torch.device('cuda' if cuda_is_available else 'cpu')

# lines_filepath = os.path.join("movie_data", "movie_lines.txt")
# convo_filepath = os.path.join("movie_data", "movie_conversations.txt")



# line_fields = ["line_id", 'character_id', "movie_id", "character", 'text']
# lines = {}

# with open(lines_filepath, 'r', encoding='iso-8859-1') as file:
#     for line in file:
#         values = line.split(" +++$+++ ")
#         lineObj = {}
#         for i , field in enumerate(line_fields):
#             lineObj[field] = values[i]
#         lines[lineObj["line_id"]] = lineObj

# convo_fields = ["character_1_id", "character_2_id", "movie_id", 'utterance_id']
# conversations = []

# with open(convo_filepath, 'r', encoding='iso-8859-1') as file:
#         for line in file:
#             values = line.split(" +++$+++ ")

#             convoObj = {}

#             for i, field in enumerate(convo_fields):
#                 convoObj[field] = values[i]

#             line_ids = eval(convoObj['utterance_id'])

#             convoObj['lines'] = []

#             for line_id in line_ids:
#                 convoObj['lines'].append(lines[line_id])
#             conversations.append(convoObj)

# qa_pairs = []

# for conversation in conversations:
#     for i in range(len(conversation['lines']) -1):
#         input_line = conversation['lines'][i]["text"].strip()
#         target_line = conversation['lines'][i + 1]['text'].strip()

#         if input_line and target_line:
#             qa_pairs.append([input_line, target_line])


# datafile = os.path.join("movie_data", "formatted_lines.txt")
# delimiter = '\t'

# delimiter = str(codecs.decode(delimiter, "unicode_escape"))

# print("\nWriting formatted file..")

# with open(datafile, 'w', encoding= 'utf-8') as outputfile:
#     writer = csv.writer(outputfile, delimiter = delimiter)

#     for pair in qa_pairs:
#         writer.writerow(pair)
# print('Done!')
