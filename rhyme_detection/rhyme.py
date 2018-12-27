'''Functions for getting rhyme scheme'''

from keras.models import load_model
from network import prepare_pairs

import argparse
import os
import pickle
import numpy as np

def does_rhyme(word1, word2, char_to_int, word_length, model):
	# Prepare a pair for the two words
	pair = prepare_pairs([word1], [word2], char_to_int, word_length)
	pair = np.array(pair[0])

	# Predict whether the pair rhymes.
	# Since the model was trained on a minibatch, we need to still
	# respect that dimension, even though we're just going to predict one point.
	pair = np.expand_dims(pair, axis=0)
	probs = model.predict(pair)[0]
	if probs[1] >= probs[0]:
		pair_rhymes = True
	else:
		pair_rhymes = False
	
	return pair_rhymes


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('language', help='Can be english.')
	args = parser.parse_args()

	# Import the model and mapping
	if args.language == 'english':
		model = load_model(os.path.join('..', 'models', 'rhyme_en.h5'))
		with open(os.path.join('..', 'models', 'char_to_int_en.pickle'), 'rb') as jar:
			char_to_int = pickle.load(jar)
		with open(os.path.join('..', 'models', 'word_length_en.pickle'), 'rb') as jar:
			word_length = pickle.load(jar)

	print(does_rhyme('sing', 'king', char_to_int, word_length, model))
	print(does_rhyme('splendor', 'render', char_to_int, word_length, model))
	print(does_rhyme('host', 'below', char_to_int, word_length, model))
	print(does_rhyme('red', 'blue', char_to_int, word_length, model))