import pickle
from conversation_data_processing import device
file = open('pairs', 'rb')
pairs = pickle.load(file)
file.close()

file = open('voc', 'rb')
voc = pickle.load(file)
file.close()
