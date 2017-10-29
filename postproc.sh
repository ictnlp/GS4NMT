#!/bin/bash

# Stanford de-tokenize for english output translation
java edu.stanford.nlp.process.PTBTokenizer -untok < $1 > $2
