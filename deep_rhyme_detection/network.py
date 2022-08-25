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

from keras.preprocessing import sequence
from keras_preprocessing.sequence import pad_sequences

from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, CSVLogger

from sklearn.model_selection import train_test_split

from tqdm import tqdm

class Corpus:
	def __init__(self, corpus_file):
		self.corpus_file = corpus_file
		self.corpus_dir = os.path.dirname(corpus_file)
		self.language = corpus_file.split('.')[0][:-2]	# 'en', 'fr', ...
		self.read_corpus()

	def read_corpus(self):
		with open(self.corpus_file, 'r') as fp:
			corpus_lines = fp.readlines()#[:5000]	# Debug
			corpus_lines = [x.split(',') for x in corpus_lines]
		self.words1 = [x[0].lower() for x in corpus_lines]
		self.words2 = [x[1].lower() for x in corpus_lines]
		self.labels = [x[2].lower() for x in corpus_lines]

		self.words = list(set(self.words1 + self.words2))
		self.concat_words = ' '.join(self.words)
		self.word_length = max(len(max(self.words1, key=len)), len(max(self.words2, key=len)))
		self.seq_length = (self.word_length * 2) + 1
		self.get_char_mapping()

	def get_char_mapping(self):
		'''
		Gets a dictionary mapping each character to a unique integer.
		'''
		chars = sorted(list(set(self.concat_words)))
		char_to_int = dict((c, i) for i, c in enumerate(chars))
		char_to_int['&'] = len(chars)	# Let & be the 'between-word' tag
		self.char_to_int = char_to_int
		self.num_chars = len(char_to_int)

	def get_char_to_int(self, word):
		int_list = []
		for char in word:
			int_list.append(self.char_to_int[char])
		return int_list

	def pad_to_length(self, word):
		while len(word) < self.word_length:
			word += ' '
		return word

	def prepare_pairs(self, verbose=False):
		'''
		Creates pairs of integer-coded padded words by character,
		separated by a '&' integer code.
		'''
		dataX = []
		if verbose:
			loop = tqdm(enumerate(self.words1))
		else:
			loop = enumerate(self.words1)
		for i, word in loop:
			word2 = self.words2[i]
			seq = self.pad_to_length(word) + '&' + self.pad_to_length(word2)
			int_seq = self.get_char_to_int(seq)
			dataX.append(int_seq)
		return dataX

	def get_onehot(self, dataX):
		'''
		Converts the integer-coded representation into a one-hot encoding.
		'''
		dataX = np.array([to_categorical(pad_sequences((data,), self.seq_length), self.num_chars + 1)  for data in dataX])
		dataX = np.array([data[0] for data in dataX])
		return dataX

	def prepare_data(self, preprocessed=False):
		if not preprocessed:
			pairs = self.prepare_pairs()
			onehot = self.get_onehot(pairs)
		else:
			with open(os.path.join(self.corpus_dir, 'rhyme_onehot_{}.pickle'.format(self.language))) as jar:
				onehot = pickle.load(jar)
		self.dataX = onehot


class Network:
	def __init__(self, corpus, num_lstm_units=8, num_epochs=10, learning_rate=0.01, batch_size=2048):
		self.corpus = corpus
		self.model_dir = os.path.join(corpus.corpus_dir, '..', 'models')

		self.num_lstm_units = num_lstm_units
		self.num_epochs = num_epochs
		self.learning_rate = learning_rate
		self.batch_size = batch_size

		self.train_test_split()

	def train_test_split(self):
		self.X_train, self.X_test, y_train, y_test = train_test_split(corpus.dataX, corpus.labels, test_size=0.2, random_state=42)
		self.y_train = to_categorical(y_train)
		self.y_test = to_categorical(y_test)
		self.input_shape = self.X_train[0].shape

	def build_network_en(self):
		model = Sequential()
		model.add(Bidirectional(LSTM(self.num_lstm_units, return_sequences=True), input_shape=self.input_shape))
		model.add(Bidirectional(LSTM(self.num_lstm_units, return_sequences=True)))
		model.add(Bidirectional(LSTM(self.num_lstm_units, return_sequences=True)))
		model.add(Bidirectional(LSTM(self.num_lstm_units, return_sequences=True)))
		model.add(Bidirectional(LSTM(self.num_lstm_units, return_sequences=True)))
		model.add(Bidirectional(LSTM(self.num_lstm_units)))
		model.add(Dense(2, activation='softmax'))
		adam = Adam(lr=self.learning_rate)
		model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
		print(model.summary())
		self.model = model

	def train_network_en(self):
		callbacks = [EarlyStopping(monitor='val_loss', patience=2), 
					 ModelCheckpoint(filepath=os.path.join(self.model_dir, 'rhyme_en.h5'), 
					 monitor='val_loss', save_best_only=True)]
		self.model.fit(self.X_train, self.y_train, 
					   batch_size=self.batch_size, epochs=self.num_epochs, 
					   validation_split=0.10, shuffle=True, callbacks=callbacks)



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train a neural network on a rhyming corpus.')
	parser.add_argument('language', help='The language of the rhyming corpus. Can be english.')
	parser.add_argument('--preprocessed', help='Loads a prestored pickle of the data.', action="store_true")
	args = parser.parse_args()

	# Set up the corpus object
	corpus_file = os.path.join('..', 'corpora', 'rhyme_corpus_1000_en.txt')
	corpus = Corpus(corpus_file)
	corpus.prepare_data(preprocessed=args.preprocessed)

	# Set network parameters - change these if retraining needed
	num_lstm_units = 16
	num_epochs = 10
	learning_rate = 0.001
	batch_size = 4096

	# Build and train network
	network = Network(corpus,
					  num_lstm_units=num_lstm_units, num_epochs=num_epochs,
					  learning_rate=learning_rate, batch_size=batch_size)
	network.build_network_en()
	network.train_network_en()


	
