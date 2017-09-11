# HMM-and-POS-Tagging-NLP

This project is an implementation of Part-of-Speech tagging by using Hidden Markov Model (HMM).

# Task 1 - Loading entire corpus (Brown corpus)

Each sentence is in the form of "word/POS tag" in the Brown corpus. (fire/nn means that the word 'fire' has the tag 'nn' –which is noun)

# Task 2 - Building HMM model

# Task 3 - Assigning the most probable tags

The HMM model which is trained in task 2 is used for the input_tokens.txt file to assign the most probable tags.

# Task 4 - Implementing Viterbi algorithm

Viterbi method finds the path with the highest probability by looking at all the possible tag sequences. The algorithm contains two steps:

1. Compute the probability of the most likely tag sequence.
2. Trace the back pointers to find the most likely tag sequence from the end to the beginning.

test_set.txt is used as input file.
