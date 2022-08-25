'''Functions for getting rhyme scheme'''

import argparse
import os
import pickle
import numpy as np
import string
import itertools

from keras.models import load_model
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences

from network import Corpus, Network
from tqdm import tqdm


class Stanza:
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

	def get_line_endings(self):
		line_endings = [line.split(' ')[-1] for line in self.text_lines]
		return line_endings

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
		for i, word in enumerate(tqdm(self.clean_words)):
			# If we already found the rhyme code for the word, skip it
			if word in rhyme_scheme.keys():
				continue

			rhyme_scheme[word] = 0
			found_rhyme = False
			for j in range(i):
				preceding_word = self.clean_words[j]
				if self.does_rhyme(preceding_word, word):
					found_rhyme = True
					if rhyme_scheme[preceding_word] == 0:
						num = 1
						while num in rhyme_scheme.values():
							num += 1
						rhyme_scheme[preceding_word] = num
					rhyme_scheme[word] = rhyme_scheme[preceding_word]
					#else:
			#if not found_rhyme:
			#	print('No rhyme found for {}'.format(word))

		self.rhyme_scheme = rhyme_scheme
		#print(self.rhyme_scheme)

	def get_rhyming_blocks(self):
		'''
		Consolidate the rhyme-coded text into blocks, where each word
		in the block shares the rhyme code. Account for line breaks.
		'''
		self.get_rhyme_scheme()
		coded_words = [(word, self.rhyme_scheme[word]) for word in self.clean_words]
		line_endings = self.get_line_endings()
		rhyme_blocks = []

		def in_ending(word, line_endings):
			word_in_ending = False
			for ending in line_endings:
				if word in ending and '\n' in ending:
					word_in_ending = True
					break
			return word_in_ending

		def combine_code(group_list, code):
			combined_words = ' ' .join([item[0] for item in group_list])
			combined_code = (combined_words, code)
			return combined_code

		for key, group in itertools.groupby(coded_words, lambda x: x[1]):	# Rhyme code
			group_list = list(group)
			if len(group_list) == 1:
				rhyme_blocks.append(group_list[0])
			elif len(group_list) > 1:
				ending_found = False
				for i, word in enumerate(group_list):
					# This may break if words in endings are repeated throughout - come back
					if in_ending(word[0], line_endings):
						before_ending = group_list[:i+1]
						after_ending = group_list[i+1:]
						if before_ending:
							rhyme_blocks.append(combine_code(before_ending, key))
						if after_ending:
							rhyme_blocks.append(combine_code(after_ending, key))
						ending_found = True
						break
				if not ending_found:
					combined_words = ' ' .join([item[0] for item in group_list])
					combined_code = (combined_words, key)
					rhyme_blocks.append(combined_code)

		#print(rhyme_blocks)
		return rhyme_blocks

	def scheme_to_orig(self, rhyme_blocks):
		'''
		Back-maps the rhyme scheme to the original text, including formatting
		with punctuation and line breaks.
		'''
		rhyme_block_words = [item[0] for item in rhyme_blocks]
		formatted_blocks = []
		counter = 0
		for i, block in enumerate(rhyme_block_words):
			length = len(block.split(' '))
			formatted_list = self.orig_words[counter:counter+length]
			formatted_string = ' '.join(formatted_list)
			formatted_block = (formatted_string, rhyme_blocks[i][1])
			formatted_blocks.append(formatted_block)
			counter += length
		return formatted_blocks

	def scheme_to_text(self, rhyme_blocks=None, stanza_num=None):
		if stanza_num:
			print('Calculating rhyme scheme for stanza {}...'.format(stanza_num+1))


		delimiters = [['{','}'], ['[',']'], ['(',')'], ['<','>'], ['/','/'],
					  ['`','`'], ['!','!'], ['#','#'], ['$','$'], ['%','%'],
					  ['^','^'], ['&','&'], ['*','*'], ['~','~'], ['?','?']]
		if not rhyme_blocks:
			rhyme_blocks = self.get_rhyming_blocks()
		rhyme_block_words = [item[0] for item in rhyme_blocks]

		# Back-map to formatted (with punctuation and line breaks) original text
		formatted_blocks = self.scheme_to_orig(rhyme_blocks)

		delimited_text = []
		for block in formatted_blocks:
			delimiter = delimiters[block[1]]
			delimited = delimiter[0] + block[0] + delimiter[1]
			# Push newlines after the delimiter
			if '\n' in delimited:
				delimited = delimited.replace('\n', '') + '\n'
			delimited_text.append(delimited)
		delimited_string = ' '.join(delimited_text)
		#print(delimited_string)
		return delimited_string

class Poem:
	'''
	Poems have multiple stanzas. This class gives functions for handling text with many
	independent stanzas or verses (rhyme schemes start over at the stanza).
	'''
	def __init__(self, full_text_lines, corpus, model):
		self.full_text_lines = full_text_lines
		self.corpus = corpus
		self.model = model
		self.get_stanzas()
		self.rhyme_blocks = None	# To be filled by below functions as needed

	def get_stanzas(self):
		'''
		Split the original text on empty lines, into stanzas/verses.
		'''
		self.stanzas_list = [list(group) for key, group in itertools.groupby(self.full_text_lines, lambda x: x == '\n') if not key]
		self.stanzas = [Stanza(stanza, self.corpus, self.model) for stanza in self.stanzas_list]

	def get_rhyme_blocks(self):
		self.rhyme_blocks = [stanza.get_rhyming_blocks() for stanza in self.stanzas]

	def get_rhyme_scheme_text(self):
		#if not self.rhyme_blocks:
		#	self.get_rhyme_blocks()
		self.rhyme_scheme = [stanza.scheme_to_text(stanza_num=i) for i, stanza in enumerate(self.stanzas)]

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('language', help='Can be english.')
	parser.add_argument('input_file', help='Path to the input file.')
	parser.add_argument('output_path', help='Path to the output directory.')
	args = parser.parse_args()

	if args.language == 'english':
		model_file = os.path.join('..', 'models', 'rhyme_en.h5')
		corpus_file = os.path.join('..', 'corpora', 'rhyme_corpus_en.txt')

	# Import the model
	print('Loading model...')
	model = load_model(model_file)

	# Set up the corpus object
	print('Setting up corpus...')
	corpus = Corpus(corpus_file)
	corpus.get_char_mapping()

	# Load in the test file
	with open(args.input_file) as fp:
		text_lines = fp.readlines()

	# Get the rhyme scheme
	print('Getting rhyme scheme.')
	poem = Poem(text_lines, corpus, model)
	poem.get_rhyme_scheme_text()
	for stanza in poem.rhyme_scheme:
		print(stanza)
	#rhyme_scheme = Stanza(text_lines, corpus, model)
	#rhyme_scheme.scheme_to_text()

	# Save to output file
	input_filename = os.path.basename(args.input_file)
	input_split = input_filename.split('.')
	output_filename = input_split[0] + '_r.' + input_split[1]
	output_file = os.path.join(args.output_path, output_filename)
	with open(output_file, 'w') as fp:
		for stanza in poem.rhyme_scheme:
			fp.write(stanza)
			fp.write('\n')