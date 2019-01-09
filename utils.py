import nltk
import pickle
import re
import numpy as np

nltk.download('stopwords')
from nltk.corpus import stopwords

# Paths for all resources for the bot.
RESOURCE_PATH = {
    'INTENT_RECOGNIZER': 'intent_recognizer.pkl',
    'TAG_CLASSIFIER': 'tag_classifier.pkl',
    'TFIDF_VECTORIZER': 'tfidf_vectorizer.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'thread_embeddings_by_tags',
    'WORD_EMBEDDINGS': 'word_embeddings.tsv',
    'WORD2ID': 'vocab.pkl'
}

unknown_symbol = '*'
start_symbol = '^'
end_symbol = '$'
padding_symbol = '#'



def text_prepare(text):
    """Performs tokenization and simple preprocessing."""
    
    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


def load_embeddings(embeddings_path):
    """Loads pre-trained word embeddings from tsv file.

    Args:
      embeddings_path - path to the embeddings file.

    Returns:
      embeddings - dict mapping words to vectors;
      embeddings_dim - dimension of the vectors.
    """
    
    # Hint: you have already implemented a similar routine in the 3rd assignment.
    # Note that here you also need to know the dimension of the loaded embeddings.
    # When you load the embeddings, use numpy.float32 type as dtype


    embeddings = {}
    entities = []
    vecs = []

    for line in open(embeddings_path):
        word, vec = line.split('\t',1)
        entities.append(word)
        vecs.append([np.float32(i) for i in vec.split('\t')])


    for i, en in enumerate(entities):
        embeddings[en] = vecs[i]



    return embeddings, 100

def question_to_vec(question, embeddings, dim):
    """Transforms a string to an embedding by averaging word embeddings."""
    
    # Hint: you have already implemented exactly this function in the 3rd assignment.


    ques2vec = np.zeros(shape=(dim), dtype=np.float32)
    n_word = 0
    for word in question.split():
      if word in embeddings:
        ques2vec += embeddings[word]
        n_word += 1
    ques2vec = ques2vec / n_word if n_word > 0 else ques2vec
    return ques2vec


def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)



def ids_to_sentence(ids, id2word):
    return [id2word[i] for i in ids]


def sentence_to_ids(sentence, word2id, padded_len):
    sentence = sentence.split()
    sent_ids = [word2id.get(word, 0) for word in sentence][:padded_len-1]+[word2id[end_symbol]]
    if len(sent_ids) < padded_len:
        sent_ids += [word2id[padding_symbol]] * (padded_len-len(sent_ids))
    sent_len = min(len(sentence)+1, padded_len)

    return sent_ids, sent_len

def batch_to_ids(sentences, word2id, max_len):
    max_len_in_batch = min(max(len(s.split()) for s in sentences)+1, max_len)
    batch_ids, batch_ids_len = [], []
    for sentence in sentences:
        ids, ids_len = sentence_to_ids(sentence, word2id, max_len_in_batch)
        batch_ids.append(ids)
        batch_ids_len.append(ids_len)
    return batch_ids, batch_ids_len
