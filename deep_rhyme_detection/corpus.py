'''
Functions for building rhyme corpus.
The idea:
- Assemble the (5000) most frequent English words.
  https://www.wordfrequency.info/
- Query a rhyming API to get lists of words that rhyme with these.
- Parse through to create pairs of words that rhyme.
- Create corresponding 'negative lists' to create pairs of words that do not rhyme.
'''

import random
import os
import argparse

from datamuse import datamuse
from tqdm import tqdm


def get_top_words(top_file, top_num=5000):
	with open(top_file, 'r') as fp:
		top_lines = fp.readlines()[:top_num]
	top_words = [x.strip() for x in top_lines]
	return top_words

def get_rhyme_dict(top_words, api=None):
	print('Getting rhyming dictionaries...')
	rhyme_dict = {}
	if not api:
		api = datamuse.Datamuse()
	for word in tqdm(top_words):
		rhymes = api.words(rel_rhy=word, max=20)
		rhymes = [x['word'] for x in rhymes]
		near_rhymes = api.words(rel_nry=word, max=20)
		near_rhymes = [x['word'] for x in near_rhymes]
		all_rhymes = rhymes + near_rhymes
		rhyme_dict[word] = all_rhymes
	return rhyme_dict

def get_neg_rhyme_dict(rhyme_dict):
	print('Getting negative rhyming dictionaries...')
	neg_rhyme_dict = {}
	for word, rhymes in tqdm(rhyme_dict.items()):
		no_rhyme = False
		while not no_rhyme:
			non_rhymes = random.choice(list(rhyme_dict.values()))
			if word not in non_rhymes:
				neg_rhyme_dict[word] = non_rhymes
				no_rhyme = True
	return neg_rhyme_dict

def get_txt(rhyme_dict, output_path, neg=False):
	print('Getting text file...')
	if neg:
		does_rhyme = 0
	else:
		does_rhyme = 1
	if os.path.exists(output_path):
		append_write = 'a' # Append if already exists
	else:
		append_write = 'w' # Create file if not
	with open(output_path, append_write) as fp:
		for word, rhymes in tqdm(rhyme_dict.items()):
			for rhyme in rhymes:
				line = '{},{},{}\n'.format(word, rhyme, does_rhyme)
				fp.write(line)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Assemble a rhyming corpus.')
	parser.add_argument('language', help='The language to build corpora for. Can be english.')
	parser.add_argument('top_num', type=int, help='The number of top words to build the corpus with (max 5000).')
	args = parser.parse_args()

	# Set up
	if args.language == 'english':
		api = datamuse.Datamuse()
		top_file = os.path.join('..', 'corpora', 'top_5000_en.csv')
		output_file = os.path.join('..', 'corpora', 'rhyme_corpus_{}_en.txt'.format(args.top_num))
	else:
		raise ValueError('Invalid corpus name: {}. Can be one of: english.'.format(args.language))
	if args.top_num > 5000:
		raise ValueError('Maximum number of top words exceeded: {}. Pick a number betwee 1 and 5000.'.format(args.top_num))

	# Get rhyming dictionaries
	top_words = get_top_words(top_file, args.top_num)
	rhyme_dict = get_rhyme_dict(top_words, api=api)
	neg_rhyme_dict = get_neg_rhyme_dict(rhyme_dict)

	# Write to text file
	get_txt(rhyme_dict, output_file)
	get_txt(neg_rhyme_dict, output_file, neg=True)


