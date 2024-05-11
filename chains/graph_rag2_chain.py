import os
import re
import spacy
import nltk

from typing import List

from chain_utils import (
    extract_questions, sample_question
)

DATA_DIR = "../data"
CHAPTERS_DIR = os.path.join(DATA_DIR, "chapters")


def extract_noun_phrases(question: str):
    tokens = nltk.word_tokenize(question)
    tagged_question = nltk.pos_tag(tokens)
    grammar = "NP: {<DT>?<JJ.*>*<NN.*>+}"
    parser = nltk.RegexpParser(grammar)
    result = parser.parse(tagged_question)
    np_leaves = [st.leaves() for st in result.subtrees()
                 if st.label() == "NP"]
    np_spans = []
    for np_leaf in np_leaves:
        np_spans.append(" ".join([s[0] for s in np_leaf]))
    return np_spans


questions = extract_questions(CHAPTERS_DIR)
cid, question = sample_question(questions)

print(question)
noun_phrases = extract_noun_phrases(question)

print(ents)
