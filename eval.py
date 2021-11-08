#!/usr/bin/env python3

# CS465 at Johns Hopkins University.
# Evaluation of taggers.
import logging
from pathlib import Path
from math import nan, exp
from typing import Counter, Tuple, Optional

import torch
from tqdm import tqdm

from corpus import Sentence, Word, EOS_WORD, BOS_WORD, OOV_WORD, TaggedCorpus, desupervise, sentence_str
from hmm import HiddenMarkovModel
from integerize import Integerizer

def model_cross_entropy(model: HiddenMarkovModel,
                        eval_corpus: TaggedCorpus) -> float:
    """Return cross-entropy per token of the model on the given evaluation corpus.
    That corpus may be either supervised or unsupervised.
    Warning: Return value is in nats, not bits."""
    with torch.no_grad():
        log_prob = 0.0
        token_count = 0
        for gold in tqdm(eval_corpus.get_sentences()):
            log_prob += model.log_prob(gold, eval_corpus).item()
            token_count += len(gold) - 1    # count EOS but not BOS
    cross_entropy = -log_prob / token_count
    logging.info(f"Cross-entropy: {cross_entropy:.4f} nats (= perplexity {exp(cross_entropy):.3f})")
    return cross_entropy

def model_error_rate(model: HiddenMarkovModel,
                     eval_corpus: TaggedCorpus,
                     known_vocab: Optional[Integerizer[Word]] = None) -> float:
    """Return the error rate of the given model on the given evaluation corpus,
    after printing cross-entropy and a breakdown of accuracy (using the logger)."""
    
    model_cross_entropy(model, eval_corpus)   # call for side effects

    with torch.no_grad():
        counts: Counter[Tuple[str, str]] = Counter()  # keep running totals here
        for gold in tqdm(eval_corpus.get_sentences()):
            predicted = model.viterbi_tagging(desupervise(gold), eval_corpus)
            counts += eval_tagging(predicted, gold, known_vocab)   # += works on dictionaries

    def fraction(c):
        num = counts['NUM',c]
        denom = counts['DENOM',c]
        return nan if denom==0 else num / denom

    categories = ['ALL', 'KNOWN', 'SEEN', 'NOVEL']
    if known_vocab is None:
        categories.remove('KNOWN')
    results = [f"{c.lower()}: {(fraction(c)):.3%}" for c in categories]            
    logging.info(f"Tagging accuracy: {', '.join(results)}")

    return 1 - fraction('ALL')  # loss value (the error rate)

def eval_tagging(predicted: Sentence, 
                 gold: Sentence, 
                 known_vocab: Optional[Integerizer[Word]]) -> Counter[Tuple[str, str]]:
    """Returns a dictionary with several performance counts,
    comparing the predicted tagging to the gold tagging of the same sentence.

    known_vocab is the words seen in the supervised corpus."""

    counts: Counter[Tuple[str, str]] = Counter()
    # print(predicted)
    # print(gold)
    for ((word, tag), (goldword, goldtag)) in zip(predicted, gold):
        #print(word, goldword)
        assert word == goldword   # sentences being compared should have the same words!
        if word is BOS_WORD or word is EOS_WORD:  # not fair to get credit for these
            continue
        if goldtag is None:                # no way to score if we don't know answer
            continue
        
        if word == OOV_WORD:                      category = 'NOVEL'
        elif known_vocab and word in known_vocab: category = 'KNOWN'
        else:                                     category = 'SEEN'    

        for c in (category, 'ALL'):
            counts['DENOM', c] += 1      # denominator of accuracy in category c
            if tag == goldtag:
                counts['NUM', c] += 1    # numerator of accuracy in category c

    return counts

def tagger_write_output(model: HiddenMarkovModel,
                        eval_corpus: TaggedCorpus,
                        output_path: Path) -> None:
    with open(output_path, 'w') as f:
        for gold in tqdm(eval_corpus.get_sentences()):
            predicted = model.viterbi_tagging(desupervise(gold), eval_corpus)
            f.write(sentence_str(predicted)+"\n")
