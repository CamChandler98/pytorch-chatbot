
import dependencies
from word_processing import Vocabulary
import pickle
def unicode_to_ascii(string):
    return ''.join(character for character in dependencies.unicodedata.normalize('NFD', string) if dependencies.unicodedata.category(character) != 'Mn' )

def normalize_string(string):
    string = unicode_to_ascii(string.lower().strip())

    string = dependencies.re.sub(r"([.!?])", r" \1", string)

    string = dependencies.re.sub(r"[^a-zA-Z.!?]+", r" ", string)

    string = dependencies.re.sub(r"\s+", r" ",string).strip()

    return string

# print(normalize_string("aa123aa!s's   dd?"))


# datafile = dependencies.os.path.join("movie_data", "formatted_lines.txt")
# print('Processing!')

# lines = open(datafile, encoding= 'utf-8').read().strip().split('\n')
# pairs = [[normalize_string(s) for s in pair.split('\t')] for pair in lines]
# print('Done!')
# for pair in pairs[:8]:
#     print(pair)

voc = Vocabulary('movie data')

max_length = 10

def filter_pair(pair):
    return  len(pair[0].split()) < max_length and len(pair[1].split()) < max_length


def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]

# pairs = [pair for pair in pairs if len(pair) > 1]
# print(f'there are {len(pairs)} in the dataset')
# pairs = filter_pairs(pairs)
# print(f'there are now {len(pairs)}')

# for pair in pairs:
#     voc.addSentence(pair[0])
#     voc.addSentence(pair[1])
#     print('counted', voc.num_words)


# min_count = 3

def trim_rare_words(voc, pairs, min_count):
    voc.trim(min_count)

    keep_pairs = []

    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]

        keep_input = True
        keep_output = True

        for word in input_sentence.split(' '):
            if word not in voc.word_to_index:
                keep_output = False
                break

        for word in output_sentence.split(' '):
            if word not in voc.word_to_index:
                keep_output = False
                break

        if keep_input and keep_output:
            keep_pairs.append(pair)

        print(f'removed {len(pairs) - len(keep_pairs)}')
    return(keep_pairs)


# pairs = trim_rare_words(voc, pairs, min_count)

# with open('pairs', 'wb') as fp:
#     print('writinging pairs')
#     pickle.dump(pairs, fp)
#     print('done')
# with open('voc', 'wb') as fp:
#     print('writing voc')
#     pickle.dump(voc, fp)
#     print('done')
