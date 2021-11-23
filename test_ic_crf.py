'''
en_crf.pkl
en_crf_raw.pkl
'''


# This file illustrates how you might experiment with the HMM interface at the prompt.
# You can also run it directly.

import logging
import math
from pathlib import Path
from typing import Callable

from corpus import TaggedCorpus, desupervise, sentence_str
from eval import eval_tagging, model_cross_entropy, model_error_rate
from hmm import HiddenMarkovModel
from crf import CRFModel
from lexicon import build_lexicon
import torch

# Set up logging
logging.basicConfig(format="%(levelname)s : %(message)s", level=logging.INFO)  # could change INFO to DEBUG
# torch.autograd.set_detect_anomaly(True)    # uncomment to improve error messages from .backward(), but slows down

# Get the corpora
icsup = TaggedCorpus(Path("../nlp6-data/icsup"), add_oov=False)
icdev = TaggedCorpus(Path("../nlp6-data/icdev"), add_oov=False)
logging.info(f"Ice cream vocabulary: {list(icsup.vocab)}")
logging.info(f"Ice cream tagset: {list(icsup.tagset)}")

# Initialize an HMM
# normal
lexicon = build_lexicon(icsup, one_hot=True, log_counts=True)  # works better with more attributes!
crf = CRFModel(icsup.tagset, icsup.vocab, lexicon, awesome=True) # not changing the name for convenience
logging.info("Running on CRF Model")

# Let's initialize with supervised training to approximately maximize the
# regularized log-likelihood.  If you want to speed this up, you can increase
# the tolerance of training (using the `tolerance` argument), since we don't 
# really have to train to convergence.
loss_sup = lambda model: model_cross_entropy(model, eval_corpus=icsup)
crf.train(corpus=icsup, loss=loss_sup, minibatch_size=30, evalbatch_size=10000, lr=0.0001, reg=1, save_path='ic_crf.pkl') 
#logging.info("sup error rate is: ", model_error_rate(crf, eval_corpus=icdev))

# Now let's throw in the unsupervised training data as well, and continue
# training to try to improve accuracy on held-out development data.
# We'll stop when accuracy is getting worse, so we can get away without regularization,
# but it would be better to search for the best `reg` and other hyperparameters in this call.
loss_dev = lambda model: model_error_rate(model, eval_corpus=icdev)
# hmm.train(corpus=entrain, loss=loss_dev, minibatch_size=30, evalbatch_size=10000, lr=0.0001, reg=0, save_path='en_crf_raw.pkl')
logging.info("dev error rate is: ", model_error_rate(crf, eval_corpus=icdev))

# More detailed look at the first 10 sentences in the held-out corpus,
# including Viterbi tagging.
for m, sentence in enumerate(icdev):
    if m >= 10: break
    viterbi = crf.viterbi_tagging(desupervise(sentence), icdev)
    counts = eval_tagging(predicted=viterbi, gold=sentence)
    num = counts['NUM', 'ALL']
    denom = counts['DENOM', 'ALL']
    
    logging.info(f"Gold:    {sentence_str(sentence)}")
    logging.info(f"Viterbi: {sentence_str(viterbi)}")
    logging.info(f"Loss:    {denom - num}/{denom}")
    logging.info(f"Prob:    {math.exp(crf.log_prob(sentence, icdev))}")
