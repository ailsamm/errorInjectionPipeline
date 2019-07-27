# Error Injection Pipeline

This pipeline takes an input txt file and injects artificially-generated errors to its text. The pipeline does so by relying on phonetic similarity to imitate the types of errors one may expect from automatic speech recognition (ASR) output. More specifically, this functions by adding

Each text to be processed within the file must be placed on a new line. Please note that the pipeline currently only functions for English text, however, this can be easily expanded by (1) downloading the [Spacy](https://spacy.io/usage/models) model in your desired language and (2) creating [Annoy](https://github.com/spotify/annoy/) dictionaries by training on a large set of text data in the same language.

This pipeline functions by first loading set of adjective, verb and noun phonetic vectors created from this [repository](https://github.com/aparrish/phonetic-similarity-vectors). These are used to find phonetically-similar alternatives to switch other words with. [Annoy](https://github.com/spotify/annoy/) is used create a phonetic vector space with which we can easily and quickly find these phonetically similar words. [Spacy](https://spacy.io/) is then used to pre-process the input text. The rest of the code randomly selects word(s) within each line of input text to switch for alternatives. The number of words switched for phonetically-similar alternatives depends on the <level> input (1-5).

## Installation

Python 3 is required to run this programme. The following installations are also required:

```bash
pip install spacy
pip install annoy
python -m spacy download en
```

## Usage

Run with:
```bash
python3 injectErrors.py --file <name_of_input_file> --level <noise_level> --splitType <split_type>
```

Note that:
* level_of_errors is an integer between 1 and 5 where 1 adds a low amount of noise and 5 adds a high amount. (Further details/specifications may be found in the [paper](http://uu.diva-portal.org/smash/record.jsf?pid=diva2%3A1321076&dswid=1942))
* split_type should be either test, train or dev - this is only used to name the output file

The output file will be output as ```<splitType>Articles_<level>.txt```in a separate folder ```errorInjectionOutput```.

## Testing
In the folder ```resources``` a file named test.txt is included. It consists of 10 sentences taken from the [Gigaword](https://github.com/harvardnlp/sent-summary) corpus. This pipeline may be tested by using this file by running the following command:
```python3 injectErrors.py --file resources/test.txt --level 2 --splitType test```
The output file for comparison will be output to ```./errorInjectionOutput/testArticles_2.txt```.

## Paper
The full paper that implements this pipeline can be found [here](http://uu.diva-portal.org/smash/record.jsf?pid=diva2%3A1321076&dswid=1942). Further details on the pipeline may also be found here.