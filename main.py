"""
Takes an input file and injects a set number of errors into the input,
outputting it as a separate file. The 'level' of errors is configurable
(see variable 'level'). This method uses phonetic similarity vectors to
switch a set number of words for phonetically-similar alternatives. Similar
vectors are calculated using Annoy.
Run with:
python3 injectErrors.py --file <name_of_input_file> --level <noise_level> --splitType <split_type>
"""
from annoy import AnnoyIndex
from optparse import OptionParser
import numpy as np
import random
import spacy


# Main function
def injectErrors():
	input_file, level, splitType = getOptions()

	# Other variables
	acceptable_pos = ["ADJ", "NOUN", "VERB"]
	failures = 0
	repFailures = 0
	sentenceFailures = 0

	# Initialises SpaCy model
	print("\n\tLoading SpaCy model...")
	model = spacy.load("en")

	initialiseAnnoy()

	name_finder = {
		"ADJ": (adj_lookup, adj_annoy, adjectives),
		"NOUN": (noun_lookup, noun_annoy, nouns),
		"VERB": (verb_lookup, verb_annoy, verbs)
	}

	# Opens new outputfiles
	output_name = "./errorInjectionOutput/" + splitType.lower() + "Articles_" + str(level) + ".txt"
	# output_name = "TEST_Errors.txt"
	output_file = open(output_name, "w")

	print("\n\tBeginning to process data in {}...\n".format(input_file))
	with open(input_file, "r") as f:
		for i, line in enumerate(f):

			replacements = []

			target_changes = level
			found = 0

			if i % 100 == 0:
				print("Now processing line {}".format(i))

			article = model(line)  # Creates SpaCy representation of the article
			# article = model(line.decode("utf-8")) # Creates SpaCy representation of the article

			random_tokens = random.sample(range(len(article)),
										  len(article))  # Creates random order of indexes to iterate through

			for ind in random_tokens:
				if str(article[ind].pos_) in acceptable_pos:  # Only continue if word is adjective, noun or verb
					try:
						word = str(article[ind]).lower().strip()
						# print("\nWord is {}".format(word))

						lookup_name = name_finder[str(article[ind].pos_)][0]
						annoy_builder = name_finder[str(article[ind].pos_)][1]
						word_list = name_finder[str(article[ind].pos_)][2]

						word_vector = lookup_name[word]

						similar_words = [word_list[i] for i in annoy_builder.get_nns_by_vector(word_vector,
																							   6)]  # List of similar words not including actual word
						# print("similar words are {}".format(similar_words))
						random_ind = random.randint(1, 5)
						replacement = (
						str(article[ind]), similar_words[random_ind])  # Stores replacement to be made as tuple
						replacements.append(replacement)
						# print(replacement)
						found += 1

						if found >= target_changes:  # Breaks when all the necessary changes have been made
							break

					except Exception:
						failures += 1  # Failures is printed at end to give an idea of how many cases failed

			if found < target_changes:
				# print("SENTENCE FAILED")
				output_file.write("FFFAILED SSSENTENCE\n")
				sentenceFailures += 1
				continue

			updated_sentence = line.lower()
			updated_sentence = updated_sentence.split(" ")

			# Actually effectuates the intended changes
			for replacement in replacements:
				try:
					index_to_replace = updated_sentence.index(replacement[0])
					updated_sentence[index_to_replace] = replacement[1]  # .upper()
				except Exception:
					# print("Replacement failure found for sentence {} in word {}".format())
					repFailures += 1

			final_sentence = " ".join(updated_sentence)

			# print("ORIGINAL Sentence is:\n{}".format(article))
			# print("FINAL Sentence is:\n{}".format(final_sentence))

			output_file.write(final_sentence)

	# Print statements to document number of errors while iterating through sentences
	print("\n----- Total number of failures: {} -----".format(failures))
	print("\n----- Total number of replacement failures: {} -----".format(repFailures))
	print("\n----- Total number of sentence failures: {} -----".format(sentenceFailures))

	output_file.close()

	return output_name


# Handles command line arguments
def getOptions():
	parser = OptionParser()
	parser.add_option("-f", "--file", dest="file",
					  help="file to process")
	parser.add_option("-l", "--level", dest="level", type="int",
					  help="level of errors to inject")
	parser.add_option("-s", "--splitType", dest="splitType",
					  help="split type: either training, test or dev. Only used for naming output files")

	(options, args) = parser.parse_args()

	if options.level == None:
		raise Exception("No level given. Please use the --level flag.")
	if options.file == None:
		raise Exception("No input file given. Please use the --file flag.")
	if options.splitType == None:
		raise Exception("No split type given. Please use the --splitType flag.")

	input_file = options.file
	level = options.level
	splitType = options.splitType

	return input_file, level, splitType

# Initialises and creates Annoy vector spaces
def initialiseAnnoy():
	adj_annoy = AnnoyIndex(50, metric='angular')
	noun_annoy = AnnoyIndex(50, metric='angular')
	verb_annoy = AnnoyIndex(50, metric='angular')
	adjectives = list()
	adj_lookup = dict()
	nouns = list()
	noun_lookup = dict()
	verbs = list()
	verb_lookup = dict()

	# Loads Annoy ADJ vectors
	print("\n\tLoading adjective vectors...")
	for i, line in enumerate(open("./vectors/adjVectors", "r")):
		line = line.strip()
		word, vec_s = line.split("  ")
		vec = [float(n) for n in vec_s.split()]
		adj_annoy.add_item(i, vec)
		adj_lookup[word] = vec
		adjectives.append(word)
	adj_annoy.build(50)

	# Loads Annoy NOUN vectors
	print("\n\tLoading noun vectors...")
	for i, line in enumerate(open("./vectors/nounVectors", "r")):
		line = line.strip()
		word, vec_s = line.split("  ")
		vec = [float(n) for n in vec_s.split()]
		noun_annoy.add_item(i, vec)
		noun_lookup[word] = vec
		nouns.append(word)
	noun_annoy.build(50)

	# Loads Annoy VERB vectors
	print("\n\tLoading verb vectors...")
	for i, line in enumerate(open("./vectors/verbVectors", "r")):
		line = line.strip()
		word, vec_s = line.split("  ")
		vec = [float(n) for n in vec_s.split()]
		verb_annoy.add_item(i, vec)
		verb_lookup[word] = vec
		verbs.append(word)
	verb_annoy.build(50)



injectErrors()