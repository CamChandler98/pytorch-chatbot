pad_token = 0
sos_token = 1
eos_token = 2

class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.word_to_index = {}
        self.word_to_count = {}
        self.index_to_word = {pad_token: 'PAD', sos_token: 'SOS', eos_token: 'EOS'}
        self.num_words = 3


    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)


    def addWord(self, word):
        if word not in self.word_to_index:
            self.word_to_index[word] = self.num_words
            self.word_to_count[word] = 1
            self.index_to_word[self.num_words] = word
            self.num_words += 1
        else:
            self.word_to_count[word] += 1


    def trim( self, min_count):
        keep_words = []

        for key, value in self.word_to_count.items():
            if value >= min_count:
                keep_words.append(key)

        self.word_to_index = {}
        self.word_to_count = {}
        self.index_to_word = {pad_token: 'PAD', sos_token: 'SOS', eos_token: 'EOS'}
        self.num_words = 3

        for word in keep_words:
            self.addWord(word)
