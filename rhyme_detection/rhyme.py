'''Functions for getting rhyme scheme'''

import argparse
import os
import pickle
import numpy as np
import string

from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

from network import Corpus, Network


class Rhymescheme:
	def __init__(self, text_lines, corpus, model):
		self.text_lines = text_lines
		self.corpus = corpus
		self.model = model
		self.load_text()

	def load_text(self):
		self.orig_words = [word for line in self.text_lines for word in line.split(' ')]
		self.clean_words = [word.strip().strip(string.punctuation).lower() for word in self.orig_words]

	def get_char_to_int(self, word):
		int_list = [corpus.char_to_int[char] for char in word]
		return int_list

	def pad_to_length(self, word):
		while len(word) < corpus.word_length:
			word += ' '
		return word

	def prepare_pair(self, word1, word2):
		seq = self.pad_to_length(word1) + '&' + self.pad_to_length(word2)
		int_seq = self.get_char_to_int(seq)
		return int_seq

	def get_onehot(self, seq):
		seq = [seq]
		onehot = np.array([to_categorical(pad_sequences((data,), corpus.seq_length), corpus.num_chars+1)  for data in seq])
		onehot = np.array([data[0] for data in onehot])
		return onehot

	def does_rhyme(self, word1, word2):
		'''
		Predicts whether a pair of words rhymes.
		'''
		pair = self.prepare_pair(word1, word2)
		onehot = self.get_onehot(pair)
		probs = self.model.predict(onehot)[0]
		if probs[1] >= probs[0]:
			pair_rhymes = True
		else:
			pair_rhymes = False
		
		return pair_rhymes

	def get_rhyme_scheme(self):
		'''
		Heuristically determines rhyme scheme this way:
		- For all words, look back at every word preceding it.
		- If the word and a preceding word rhyme, code them together.
		'''
		rhyme_scheme = {}
		for i, word in enumerate(self.clean_words):
			if word in rhyme_scheme.keys():
				# We have already assigned a rhyme to this word
				continue

			rhyme_scheme[word] = 0
			found_rhyme = False
			for j in range(i):
				preceding_word = self.clean_words[j]
				if self.does_rhyme(preceding_word, word):
					found_rhyme = True
					#print('{} rhymes with {}'.format(word, preceding_word))
					#if preceding_word != word:
					if rhyme_scheme[preceding_word] == 0:
						num = 1
						while num in rhyme_scheme.values():
							num += 1
						rhyme_scheme[preceding_word] = num
					rhyme_scheme[word] = rhyme_scheme[preceding_word]
					#else:
			if not found_rhyme:
				print('No rhyme found for {}'.format(word))

		self.rhyme_scheme = rhyme_scheme
		print(self.clean_words)
		print(self.rhyme_scheme)




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('language', help='Can be english.')
	args = parser.parse_args()

	if args.language == 'english':
		model_file = os.path.join('..', 'models', 'rhyme_en.h5')
		corpus_file = os.path.join('..', 'corpora', 'rhyme_corpus_en.txt')

	# Import the model
	model = load_model(model_file)

	# Set up the corpus object
	corpus = Corpus(corpus_file)
	corpus.get_char_mapping()

	# Load in the test file
	with open(os.path.join('..', 'test_files', 'deck_thyself.txt')) as fp:
		text_lines = fp.readlines()

	# Get the rhyme scheme
	rhyme_scheme = Rhymescheme(text_lines, corpus, model)
	rhyme_scheme.get_rhyme_scheme()