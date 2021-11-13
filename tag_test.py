#!/usr/bin/env python3

# 601.465/665 - Natural Language Processing
# Assignment 6 - HMM
# Author: Amir Hussein

import argparse
import logging
import math
import pdb
from pathlib import Path
from typing import Callable
from corpus import TaggedCorpus, desupervise, sentence_str
from eval import eval_tagging, model_cross_entropy, model_error_rate, tagger_write_output
from hmm import HiddenMarkovModel
from lexicon import build_lexicon
import pdb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "eval_file",
        type=Path,
        nargs=1,
        help="path to the evaluation file"
    )

    parser.add_argument(
        "--model",
        type=Path,
        default = None,
        help="path to the model file"
    )
    parser.add_argument(
        "--train",
        type=Path,
        nargs="*",
        default = None,
        help="path to the training_files"
    )
    return parser.parse_args()

def main():
    # Set up logging
    logging.basicConfig(format="%(levelname)s : %(message)s", level=logging.INFO)  # could change INFO to DEBUG
    args = parse_args()
    # load the training data
    #pdb.set_trace()
    if args.train != None:
        if len(args.train)>1:
            # Semi-supervised
            entrain = TaggedCorpus(Path(args.train[0]), Path(args.train[1]) )                              # all training
            endev = TaggedCorpus(Path(args.eval_file[0]), tagset=entrain.tagset, vocab=entrain.vocab)  # evaluation
            ensup =   TaggedCorpus(Path(args.train[0]), tagset=entrain.tagset, vocab=entrain.vocab)  # supervised training
            
            logging.info(f"Tagset: f{list(entrain.tagset)}")
            known_vocab = TaggedCorpus(Path(args.train[0])).vocab    # words seen with supervised tags; used in evaluation

            
            # Initialize an HMM
            #lexicon = build_lexicon(ensup, one_hot=True)   # one-hot lexicon: separate parameters for each word
            #lexicon = build_lexicon(entrain, embeddings_file=Path('../lexicons/words-50.txt'))  # works better with more attributes!
            lexicon = build_lexicon(entrain, embeddings_file=Path('words-50.txt'))
            hmm = HiddenMarkovModel(entrain.tagset, entrain.vocab, lexicon)
            if args.model != None:
                hmm = hmm.load(args.model)
            # supervised training 
            loss_sup = lambda model: model_cross_entropy(model, eval_corpus=ensup)
            hmm.train(corpus=ensup, loss=loss_sup, minibatch_size=30, evalbatch_size=10000, lr=0.0001, reg=1) 
            # continue with unsupervised
            loss_dev = lambda model: model_error_rate(model, eval_corpus=endev, known_vocab=known_vocab)
            hmm.train(corpus=entrain, loss=loss_dev, minibatch_size=30, evalbatch_size=10000, lr=0.0001, reg=0)

        else:
            # supervised training
            ensup =   TaggedCorpus(Path(args.train[0]))  # supervised training
            endev = TaggedCorpus(Path(args.eval_file[0]), tagset=ensup.tagset, vocab=ensup.vocab)  # evaluation

            # Initialize an HMM
            #lexicon = build_lexicon(ensup, one_hot=True)   # one-hot lexicon: separate parameters for each word
            #lexicon = build_lexicon(entrain, embeddings_file=Path('../lexicons/words-50.txt'))  # works better with more attributes!
            lexicon = build_lexicon(ensup, embeddings_file=Path('words-50.txt'))
            hmm = HiddenMarkovModel(ensup.tagset, ensup.vocab, lexicon)

            loss_sup = lambda model: model_cross_entropy(model, eval_corpus=ensup)
            if args.model != None:
                hmm = hmm.load(args.model)
            hmm.train(corpus=ensup, loss=loss_sup, minibatch_size=30, evalbatch_size=10000, lr=0.0001, reg=1) 


    else:
        # evaluate without training
        endev = TaggedCorpus(Path(args.eval_file[0]))  # evaluation
         # Initialize an HMM
        #lexicon = build_lexicon(endev, one_hot=True)   # one-hot lexicon: separate parameters for each word
        #lexicon = build_lexicon(entrain, embeddings_file=Path('../lexicons/words-50.txt'))  # works better with more attributes!
        lexicon = build_lexicon(endev, embeddings_file=Path('words-50.txt'))
        hmm = HiddenMarkovModel(endev.tagset, endev.vocab, lexicon)
        assert args.model != None
        hmm = hmm.load(args.model)
        endev = TaggedCorpus(Path(args.eval_file[0]),tagset=hmm.tagset, vocab=hmm.vocab)  # evaluation
    # for m, sentence in enumerate(endev):
    #     viterbi = hmm.viterbi_tagging(desupervise(sentence), endev)
    #     counts = eval_tagging(predicted=viterbi, gold=sentence, 
    #                         known_vocab=known_vocab)
    #     num = counts['NUM', 'ALL']
    #     denom = counts['DENOM', 'ALL']
        
    #     logging.info(f"Gold:    {sentence_str(sentence)}")
    #     logging.info(f"Viterbi: {sentence_str(viterbi)}")
    #     logging.info(f"Loss:    {denom - num}/{denom}")
    #     logging.info(f"Prob:    {math.exp(hmm.log_prob(sentence, endev))}")

    tagger_write_output(hmm, endev, Path("%s.output" %str(args.eval_file[0])))
if __name__ == "__main__":
    main()
