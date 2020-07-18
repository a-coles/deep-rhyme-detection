# deep-rhyme-detection

This software is designed to detect rhyme and rhyme schemes, especially in the context of poetry or lyrics. It trains and uses a recurrent neural network in order to predict which parts of the text rhyme with each other. This repository also contains some scripts for the assembly of rhyming corpora and for rendering of the text with rhyme scheme included.

#### Example

| Input | Output |
| --- | --- |
| `My song is love unknown,` | `/My/ {song} (is) [love] <unknown,>` |
| `my Savior's love to me;` | `/my/ #Savior's# [love to me;]` |
| `love to the loveless shown,` | `[love to the loveless] <shown,>` |
| `that they might lovely be.` | `!that! <they> *might* <lovely> [be.]` |
| `O who am I, that for my sake,` | `[O who] *am I,* !that! {for} /my/ !sake,!` |
| `my Lord should take frail flesh and die?` | `/my/ {Lord} [should] !take! #frail flesh# /and die?/` |


#### Supported languages

* English
* French _(coming soon)_


## Installation

To install, first clone this repo:

```
cd path/to/desired/install/location
git clone https://github.com/a-coles/deep-rhyme-detection.git
```

Then install dependencies by running:

```
pip install -r requirements.txt
```

## Use

This repo is designed to be easily integratable with other projects, with functions and classes sorted into modules. To see how it works, though, you can run the top-level script in the main of `rhyme.py` like this:

```
cd deep_rhyme_detection/
python rhyme.py [language] [input_file] [output_dir] [format]
```

where the arguments are:

* `language`, the language of the text you want to analyze. This can only currently take the value `english` (one day there will hopefully be support for other languages). 
* `input_file`, the path to the file containing the text to analyze. This should be a simple text file where lines in a stanza are separated by line breaks and stanzas are separated by a blank line. Punctuation and the like is fine to include.
* `output_dir`, the path to the directory where the analyzed text will be dumped to a new text file.
* `format`, whether the rhyme scheme annotated output should be `txt` or `html`.



## Other scripts

### Corpus building

This repo comes with a pre-created corpus of English rhyming data, built from [WordFrequency's list of most frequent English words](https://www.wordfrequency.info/free.asp) and the [Datamuse Python API](https://github.com/gmarmstrong/python-datamuse/). The corpus is constructed by querying Datamuse for words that rhyme with each frequent English word and generating pairs of rhyming and non-rhyming words from these results.

If you would like to create a corpus yourself, you can do so like this:

```
cd deep_rhyme_detection/
python corpus.py [language] [top_num]
```

where the arguments are:

* `language`, the language for which you want to build a corpus. This can only currently take the value `english` (one day there will hopefully be support for other languages). 
* `top_num`, an integer representing the number of frequent words to use, with maximum 5000. A too-large value may introduce too much noise into the data.

### Network training

This repo comes with a pretrained RNN that predicts whether two English words rhyme with each other, `rhyme_en.h5` under the `models/` directory. The English network could always use more tuning (pull requests welcome!), but for now, it consists of a 6-layer bidirectional character-level LSTM, each with 16 units, using an Adam optimizer and a cross-entropy loss.

If you would like to re-train the RNN with different hyperparameters, you can adjust their values in the main method of `network.py`. Open the `network.py` file in a text editor and change the values in the main as you see fit:

```
# Set network parameters - change these if retraining needed
num_lstm_units = 16
num_epochs = 10
learning_rate = 0.001
batch_size = 4096
```

Then, from the command line, run:

```
python network.py [language] [--preprocessed]
```

where the arguments are:

* `language`, the language of the training corpus. This can only currently take the value `english` (one day there will hopefully be support for other languages). 
* `--preprocessed`, a switch you should only set if you have an offline `pickle` of your training corpus in the onehot format created by the `Corpus` class.
