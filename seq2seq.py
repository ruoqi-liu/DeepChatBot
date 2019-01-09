import tensorflow as tf
import tensorflow.contrib as contrib
from utils import *
from sklearn.model_selection import train_test_split
import numpy as np
import random
from sklearn.metrics import mean_absolute_error

data = unpickle_file('generate_movie_dialogue.pkl')
unknown_symbol = '*'
start_symbol = '^'
end_symbol = '$'
padding_symbol = '#'
word2id, id2word = unpickle_file('vocab.pkl')


def split_dataset():
    train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
    return train_set, test_set


def generate_vocab():
    word_freq = {}
    unique_set = set()
    word2id = {}
    id2word = {}
    for input, output in data:
        for word in input.split():
            word_freq[word] = word_freq.get(word, 0) + 1
        for word in output.split():
            word_freq[word] = word_freq.get(word, 0) + 1
    for word, freq in word_freq.items():
        if freq > 0:
            unique_set.add(word)

    for i, symbol in enumerate('#^$'):
        word2id[symbol] = i
        id2word[i] = symbol

    for i, word in enumerate(unique_set):
        word2id[word] = i+3
        id2word[i+3] = word

    pickle.dump((word2id, id2word), open('vocab.pkl', 'wb'))


def sentence_to_ids(sentence, word2id, padded_len):
    sentence = sentence.split()
    sent_ids = [word2id.get(word, 0) for word in sentence][:padded_len-1]+[word2id[end_symbol]]
    if len(sent_ids) < padded_len:
        sent_ids += [word2id[padding_symbol]] * (padded_len-len(sent_ids))
    sent_len = min(len(sentence)+1, padded_len)

    return sent_ids, sent_len


def ids_to_sentence(ids, id2word):
    return [id2word[i] for i in ids]


def batch_to_ids(sentences, word2id, max_len):
    max_len_in_batch = min(max(len(s.split()) for s in sentences)+1, max_len)
    batch_ids, batch_ids_len = [], []
    for sentence in sentences:
        ids, ids_len = sentence_to_ids(sentence, word2id, max_len_in_batch)
        batch_ids.append(ids)
        batch_ids_len.append(ids_len)
    return batch_ids, batch_ids_len


def generate_batches(samples, batch_size=64):
    X, Y = [], []
    for i, (x,y) in enumerate(samples, 1):
        X.append(x)
        Y.append(y)
        if i % batch_size == 0:
            yield X, Y
            X, Y = [], []

    if X and Y:
        yield X, Y


class Seq2SeqModel(object):
    def __declare_placeholders(self):
        self.input_batch = tf.placeholder(shape=(None, None), dtype=tf.int32, name='input_batch')
        self.input_batch_lengths = tf.placeholder(shape=(None, ), dtype=tf.int32, name='input_batch_lengths')

        self.ground_truth = tf.placeholder(shape=(None, None), dtype=tf.int32, name='ground_truth')
        self.ground_truth_lengths = tf.placeholder(shape=(None, ), dtype=tf.int32, name='ground_truth_length')

        self.dropout_ph = tf.placeholder_with_default(tf.cast(1.0, tf.float32), shape=[])
        self.learning_rate_ph = tf.placeholder(shape=[], dtype=tf.float32)

    def __creare_embeddings(self, vocab_size, embeddings_size):
        random_initializer = tf.random_uniform((vocab_size, embeddings_size), -1.0, 1.0)
        self.embeddings = tf.Variable(random_initializer, dtype=tf.float32, name='embeddings')
        self.input_batch_embedded = tf.nn.embedding_lookup(self.embeddings, self.input_batch)

    def __build_encoder(self, hidden_size):
        encoder_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(hidden_size), input_keep_prob=self.dropout_ph)
        self.encoder_outputs, self.final_encoder_state = tf.nn.dynamic_rnn(
            encoder_cell,
            inputs=self.input_batch_embedded,
            sequence_length=self.input_batch_lengths,
            dtype=tf.float32)

    def __build_decoder(self, hidden_size, vocab_size, max_iter, start_symbol_id, end_symbol_id):
        batch_size = tf.shape(self.input_batch)[0]
        start_tokens = tf.fill([batch_size], start_symbol_id)
        ground_truth_as_input = tf.concat([tf.expand_dims(start_tokens, 1), self.ground_truth], 1)

        self.ground_truth_embedded = tf.nn.embedding_lookup(self.embeddings, ground_truth_as_input)

        train_helper = contrib.seq2seq.TrainingHelper(self.ground_truth_embedded, self.ground_truth_lengths)

        infer_helper = contrib.seq2seq.GreedyEmbeddingHelper(self.embeddings, start_tokens, end_symbol_id)

        def decode(helper, scope, reuse=None):
            with tf.variable_scope(scope, reuse=reuse):
                # decoder_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(hidden_size, reuse=reuse), input_keep_prob=self.dropout_ph)
                # decoder_cell = contrib.rnn.OutputProjectionWrapper(decoder_cell, vocab_size, reuse=reuse)
                # decoder = contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=helper, initial_state=self.final_encoder_state)
                # outputs, _, _ = contrib.seq2seq.dynamic_decode(decoder=decoder, maximum_iterations=max_iter,
                #                                                   output_time_major=False, impute_finished=True)


                memory = self.encoder_outputs
                attention_mechanism = contrib.seq2seq.BahdanauAttention(
                    num_units=hidden_size, memory=memory,
                    memory_sequence_length=self.input_batch_lengths)
                decoder_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(hidden_size, reuse=reuse),
                                                             input_keep_prob=self.dropout_ph)
                attn_cell = contrib.seq2seq.AttentionWrapper(
                    decoder_cell, attention_mechanism, attention_layer_size=hidden_size)
                out_cell = contrib.rnn.OutputProjectionWrapper(attn_cell, vocab_size, reuse=reuse)
                decoder = contrib.seq2seq.BasicDecoder(
                    cell=out_cell, helper=helper,
                    initial_state=out_cell.zero_state(dtype=tf.float32, batch_size=batch_size))

                outputs, _, _ = contrib.seq2seq.dynamic_decode(decoder=decoder, maximum_iterations=max_iter,
                                                               output_time_major=False, impute_finished=True)

            return outputs

        self.train_outputs = decode(train_helper, 'decode')
        self.infer_outputs = decode(infer_helper, 'decode', reuse=True)

    def __compute_loss(self):
        weights = tf.cast(tf.sequence_mask(self.ground_truth_lengths), dtype=tf.float32)
        self.loss = contrib.seq2seq.sequence_loss(
            logits=self.train_outputs.rnn_output, targets=self.ground_truth, weights=weights)

    def __perform_optimization(self):
        self.train_op = contrib.layers.optimize_loss(
            loss=self.loss,
            global_step=tf.train.get_global_step(),
            learning_rate=self.learning_rate_ph,
            optimizer='Adam',
            clip_gradients=1.0)

    def __init__(self, vocab_size, embeddings_size, hidden_size, max_iter, start_symbol_id, end_symbol_id, padding_symbol_id):
        self.__declare_placeholders()
        self.__creare_embeddings(vocab_size, embeddings_size)
        self.__build_encoder(hidden_size)
        self.__build_decoder(hidden_size, vocab_size, max_iter, start_symbol_id, end_symbol_id)

        self.__compute_loss()
        self.__perform_optimization()

        self.train_predictions = self.train_outputs.sample_id
        self.infer_predictions = self.infer_outputs.sample_id

    def train_on_batch(self, session, X, X_seq_len, Y, Y_seq_len, learning_rate, dropout_keep_probability):
        feed_dict = {
            self.input_batch:X,
            self.input_batch_lengths:X_seq_len,
            self.ground_truth:Y,
            self.ground_truth_lengths:Y_seq_len,
            self.learning_rate_ph:learning_rate,
            self.dropout_ph:dropout_keep_probability
        }
        pred, loss, _ = session.run([
            self.train_predictions,
            self.loss,
            self.train_op
        ], feed_dict=feed_dict)

        return pred, loss

    def predict_for_batch(self, session, X, X_seq_len):
        feed_dict={
            self.input_batch:X,
            self.input_batch_lengths:X_seq_len
        }
        pred=session.run([self.infer_predictions], feed_dict=feed_dict)[0]
        return pred

    def predict_for_batch_with_loss(self, session, X, X_seq_len, Y, Y_seq_len):
        feed_dict={
            self.input_batch:X,
            self.input_batch_lengths:X_seq_len,
            self.ground_truth: Y,
            self.ground_truth_lengths: Y_seq_len
        }
        pred, loss = session.run([
            self.infer_predictions,
            self.loss
        ], feed_dict=feed_dict)
        return pred, loss




if __name__ == '__main__':
    train_set, test_set = split_dataset()


    tf.reset_default_graph()
    model = Seq2SeqModel(
        vocab_size=len(word2id), embeddings_size=20, max_iter=8, hidden_size=512,
        start_symbol_id=word2id[start_symbol], end_symbol_id=word2id[end_symbol], padding_symbol_id=word2id[padding_symbol])
    batch_size = 128
    n_epochs = 30
    learning_rate = 0.001
    dropout_keep_probability = 0.9
    max_len = 20
    n_step = int(len(train_set)/batch_size)

    export_path = './savedmodel'
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    session = tf.Session()
    session.run(tf.global_variables_initializer())


    # all_model_predictions = []
    # all_ground_truth = []

    print('Start training... \n')
    for epoch in range(n_epochs):
        random.shuffle(train_set)
        random.shuffle(test_set)
        print('Train: epoch', epoch + 1)
        for n_iter, (X_batch, Y_batch) in enumerate(generate_batches(train_set, batch_size=batch_size)):

            X_batch, lx = batch_to_ids(X_batch, word2id, max_len)
            Y_batch, ly = batch_to_ids(Y_batch, word2id, max_len)

            predictions, loss = model.train_on_batch(session, X_batch, lx, Y_batch, ly, learning_rate,
                                                     dropout_keep_probability)

            if n_iter % 50 == 0:
                print("Epoch: [%d/%d], step: [%d/%d], loss: %f" % (epoch + 1, n_epochs, n_iter + 1, n_step, loss))



        X_sent, Y_sent = next(generate_batches(test_set, batch_size=batch_size))
        X, lx = batch_to_ids(X_sent, word2id, max_len)
        Y, ly = batch_to_ids(Y_sent, word2id, max_len)
        predictions, loss = model.predict_for_batch_with_loss(session, X, lx, Y, ly)
        print('Test: epoch', epoch + 1, 'loss:', loss, )
        for x, y, p in list(zip(X, Y, predictions))[:3]:
            print('X:', ' '.join(ids_to_sentence(x, id2word)))
            print('Y:', ' '.join(ids_to_sentence(y, id2word)))
            print('O:', ' '.join(ids_to_sentence(p, id2word)))
            print('')

        # model_predictions = []
        # ground_truth = []
        #
        # for X_batch, Y_batch in generate_batches(test_set, batch_size=batch_size):
        #
        #     X_batch, lx = batch_to_ids(X_batch, word2id, max_len)
        #     Y_batch, ly = batch_to_ids(Y_batch, word2id, max_len)
        #     pre = model.predict_for_batch(session, X_batch, lx)
        #
        #     for y, p in zip(Y_batch, pre):
        #         y_sent = ' '.join(ids_to_sentence(y, id2word))
        #         y_sent = y_sent[:y_sent.find('$')]
        #         p_sent = ' '.join(ids_to_sentence(p, id2word))
        #         p_sent = p_sent[:p_sent.find('$')]
        #
        #         model_predictions.append(int(p_sent))
        #         ground_truth.append(int(y_sent))
        #
        # all_model_predictions.append(model_predictions)
        # all_ground_truth.append(ground_truth)

    print('\n...training finished.')

    # Save the variables to disk.
    # inputs = {
    #     "input_batch": model.input_batch,
    #     "input_batch_lengths": model.input_batch_lengths
    # }
    # outputs = {"prediction": model.infer_predictions}
    # tf.saved_model.simple_save(
    #     session, "./seq2seq_model", inputs, outputs)

    # save the model


    input_batch = tf.saved_model.utils.build_tensor_info(model.input_batch)
    input_batch_lengths = tf.saved_model.utils.build_tensor_info(model.input_batch_lengths)
    prediction = tf.saved_model.utils.build_tensor_info(model.infer_predictions)

    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs = {
                "input_batch": input_batch,
                "input_batch_lengths": input_batch_lengths
            },
            outputs={"prediction": prediction},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    builder.add_meta_graph_and_variables(
        session, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                prediction_signature
        },
    )
    builder.save()


    x_b, l = batch_to_ids(['how are you', 'thank you', 'i love you', 'what is your name?'], word2id, max_len)
    pre = model.predict_for_batch(session, x_b, l)
    for x, p in zip(x_b, pre):
        pre_sent = ' '.join(ids_to_sentence(p, id2word))
        pre_sent = pre_sent[:pre_sent.find('$')]
        print(pre_sent)













