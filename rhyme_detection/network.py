'''
Neural network functions
'''
import os
import numpy as np
import pickle
import argparse

from keras.utils import to_categorical, np_utils
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional
from keras.layers.embeddings import Embedding

from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot

from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, CSVLogger

from sklearn.model_selection import train_test_split

from tqdm import tqdm

def get_char_mapping(concat_words):
	chars = sorted(list(set(concat_words)))
	char_to_int = dict((c, i) for i, c in enumerate(chars))
	# Let & be the 'between-word' tag
	char_to_int['&'] = len(chars)
	return char_to_int

def get_char_to_int(char_to_int, word):
	int_list = []
	for char in word:
		int_list.append(char_to_int[char])
	return int_list

def pad_to_length(word, word_length):
	while len(word) < word_length:
		word += ' '
	return word

def prepare_pairs(words1, words2, char_to_int, word_length, verbose=False):
	# Let the word length be the length of the longest string in the word lists
	dataX = []
	if verbose:
		loop = tqdm(enumerate(words1))
	else:
		loop = enumerate(words1)
	for i, word in loop:
		word2 = words2[i]
		seq = pad_to_length(word, word_length) + '&' + pad_to_length(word2, word_length)
		int_seq = get_char_to_int(char_to_int, seq)
		dataX.append(int_seq)
	return dataX

def build_network_en(num_chars, input_shape, num_lstm_units):
	embedding_vector_length = 150
	model = Sequential()
	model.add(Embedding(num_chars, embedding_vector_length, input_length=input_shape[0]))
	model.add(Bidirectional(LSTM(num_lstm_units)))
	model.add(Dense(2, activation='softmax'))
	adam = Adam(lr=0.01)
	model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
	print(model.summary())
	return model

def train_network_en(X_train, y_train, model, num_epochs, batch_size):
	print(batch_size)
	callbacks = [EarlyStopping(monitor='val_loss', patience=2), ModelCheckpoint(filepath=os.path.join('..', 'models', 'rhyme_en.h5'), monitor='val_loss', save_best_only=True)]
	model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_split=0.10, shuffle=True, callbacks=callbacks)
	return model



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train a neural network on a rhyming corpus.')
	parser.add_argument('language', help='The language of the rhyming corpus. Can be english.')
	parser.add_argument('--preprocessed', help='Loads a prestored pickle of the data.', action="store_true")
	args = parser.parse_args()

	corpus_file = os.path.join('..', 'corpora', 'rhyme_corpus_en.txt')
	with open(corpus_file, 'r') as fp:
		corpus_lines = fp.readlines()#[:5000]	# Debug
		corpus_lines = [x.split(',') for x in corpus_lines]
	words1 = [x[0].lower() for x in corpus_lines]
	words2 = [x[1].lower() for x in corpus_lines]
	labels = [x[2].lower() for x in corpus_lines]
	words = list(set(words1 + words2))
	concat_words = ' '.join(words)
	word_length = max(len(max(words1, key=len)), len(max(words2, key=len)))
	seq_length = (word_length * 2) + 1

	if args.preprocessed:
		if args.language == 'english':
			with open(os.path.join('..', 'corpora', 'rhyme_en.pickle'), 'rb') as jar:
				dataX = pickle.load(jar)
			with open(os.path.join('..', 'models', 'char_to_int_en.pickle'), 'rb') as jar:
				char_to_int = pickle.load(jar)
			num_chars = len(char_to_int)
			with open(os.path.join('..', 'models', 'word_length_en.pickle'), 'rb') as jar:
				word_length = pickle.load(jar)
	else:
		# Get character-to-int mapping
		char_to_int = get_char_mapping(concat_words)
		num_chars = len(char_to_int)
		with open(os.path.join('..', 'models', 'char_to_int_en.pickle'), 'wb') as jar:
			pickle.dump(char_to_int, jar, protocol=2)
		with open(os.path.join('..', 'models', 'word_length_en.pickle'), 'wb') as jar:
			pickle.dump(word_length, jar, protocol=2)

		# Get word pairs
		dataX = prepare_pairs(words1, words2, char_to_int, word_length)
		print('Integer encoding:')
		dataX = np.array(dataX)
		print(dataX)
		with open(os.path.join('..', 'corpora', 'rhyme_en.pickle'), 'wb') as jar:
			pickle.dump(dataX, jar, protocol=2)

	# Split into train/test
	print('Splitting into train/test.')
	X_train, X_test, y_train, y_test = train_test_split(dataX, labels, test_size=0.2, random_state=42)
	y_train = to_categorical(y_train)
	y_test = to_categorical(y_test)
	input_shape = X_train[0].shape
	print(X_train[0])
	print(X_train.shape)

	# Build and train network
	num_lstm_units = 8
	model = build_network_en(num_chars, input_shape, num_lstm_units)
	num_epochs = 10
	batch_size = 2048
	model = train_network_en(X_train, y_train, model, num_epochs, batch_size)


	
