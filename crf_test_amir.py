#!/usr/bin/env python3

# CS465 at Johns Hopkins University.
# Implementation of CRF Models.

from __future__ import annotations
import logging
from math import inf, log, exp, sqrt
from pathlib import Path
from typing import Callable, List, Optional, Tuple
from logsumexp_safe import logsumexp_new
import numpy as np
import torch
from torch import Tensor, nn, tensor
from torch.nn import functional as F
from tqdm import tqdm
#from scipy.special import logsumexp
from corpus import (BOS_TAG, BOS_WORD, EOS_TAG, EOS_WORD, Sentence, Tag,
                    TaggedCorpus, Word, desupervise)
from integerize import Integerizer
import pdb
# Set the seed for random numbers in torch, for replicability
torch.manual_seed(1337)
torch.cuda.manual_seed(69_420)  # No-op if CUDA isn't available

###
# CRF tagger
###
class CRFModel(nn.Module):
    """An implementation of an CRF, whose emission probabilities are
    parameterized using the word embeddings in the lexicon.
    
    We'll refer to the CRF states as "tags" and the CRF observations 
    as "words."
    """

    def __init__(self, 
                 tagset: Integerizer[Tag],
                 vocab: Integerizer[Word],
                 lexicon: Tensor,
                 unigram=False,
                 awesome=False,
                 birnn=False): 
        """Construct an CRF with initially random parameters, with the
        given tagset, vocabulary, and lexical features.
        """

        super().__init__()

        # We'll use the variable names that we used in the reading handout, for
        # easy reference.  (It's typically good practice to use more descriptive names.)
        # As usual in Python, attributes starting with _ are intended as private;
        # in this case, they might go away if you changed the parametrization of the model.

        # We omit EOS_WORD and BOS_WORD from the vocabulary, as they can never be emitted.
        # See the reading handout section "Don't guess when you know."

        assert vocab[-2:] == [EOS_WORD, BOS_WORD]  # make sure these are the last two

        self.k = len(tagset)       # number of tag types
        self.V = len(vocab) - 2    # number of word types (not counting EOS_WORD and BOS_WORD)
        self.d = lexicon.size(1)   # dimensionality of a word's embedding in attribute space
        self.unigram = unigram     # do we fall back to a unigram model?
        self.awesome = awesome     # further improvements
        self.birnn = birnn
        self.max_sen_len = 100
        self.tagset = tagset
        self.vocab = vocab
        self._E = lexicon[:-2]  # embedding matrix; omits rows for EOS_WORD and BOS_WORD
        self._E2 = lexicon

        # Useful constants that are invoked in the methods
        self.bos_t: Optional[int] = tagset.index(BOS_TAG)
        self.eos_t: Optional[int] = tagset.index(EOS_TAG)
        assert self.bos_t is not None    # we need this to exist
        assert self.eos_t is not None    # we need this to exist
        self.eye: Tensor = torch.eye(self.k)  # identity matrix, used as a collection of one-hot tag vectors
        if self.birnn:
            self.h_dim = 20
            self.f_dim = 10
        self.init_params()     # create and initialize params


    @property
    def device(self):
        """Get the GPU (or CPU) our code is running on."""
        # Why the hell isn't this already in PyTorch?
        return next(self.parameters()).device


    def _integerize_sentence(self, sentence: Sentence, corpus: TaggedCorpus) -> List[Tuple[int,Optional[int]]]:
        """Integerize the words and tags of the given sentence, which came from the given corpus."""

        # Make sure that the sentence comes from a corpus that this CRF knows
        # how to handle.
        if corpus.tagset != self.tagset or corpus.vocab != self.vocab:
            raise TypeError("The corpus that this sentence came from uses a different tagset or vocab")

        # If so, go ahead and integerize it.
        return corpus.integerize_sentence(sentence)

    def init_params(self):
        """Initialize params to small random values (which breaks ties in the fully unsupervised case).
        However, we initialize the BOS_TAG column of _WA to -inf, to ensure that
        we have 0 probability of transitioning to BOS_TAG (see "Don't guess when you know").
        See the "Parametrization" section of the reading handout."""

        # See the reading handout section "Parametrization.""

        if not self.birnn:
            ThetaB = 0.01*torch.rand(self.k, self.d)    
            self._ThetaB = nn.Parameter(ThetaB)    # params used to construct emission matrix

            WA = 0.01*torch.rand(1 if self.unigram # just one row if unigram model
                                else self.k,      # but one row per tag s if bigram model
                                self.k)           # one column per tag t
            WA[:, self.bos_t] = -inf               # correct the BOS_TAG column
            self._WA = nn.Parameter(WA)            # params used to construct transition matrix
        if self.birnn:
            self.ThetaA = nn.Parameter(torch.rand(self.f_dim, 1))
            self.ThetaB = nn.Parameter(torch.rand(self.f_dim, 1))
            self.M = nn.Parameter(torch.rand((self.h_dim, 1 + self.h_dim + self.d)))
            self.M_prime = nn.Parameter(torch.rand((self.h_dim, 1 + self.h_dim + self.d)))
            self.UA = nn.Parameter(torch.rand((self.f_dim, 1 + 2*self.h_dim + 2*self.k)))
            self.UB = nn.Parameter(torch.rand((self.f_dim, 1 + 2*self.h_dim + self.k + self.d)))

    def params_L2(self) -> Tensor:
        """What's the L2 norm of the current parameter vector?
        We consider only the finite parameters."""
        l2 = tensor(0.0)
        for x in self.parameters():
            x_finite = x[x.isfinite()]
            l2 = l2 + x_finite @ x_finite   # add ||x_finite||^2
        return l2

    def updateAB(self) -> None:
        """Set the transition and emission matrices A and B, based on the current parameters.
        See the "Parametrization" section of the reading handout."""
       # pdb.set_trace()
        # A = F.softmax(self._WA, dim=1)       # run softmax on params to get transition distributions
                                              # note that the BOS_TAG column will be 0, but each row will sum to 1
        A = self._WA

        if self.unigram:
            # A is a row vector giving unigram probabilities p(t).
            # We'll just set the bigram matrix to use these as p(t | s)
            # for every row s.  This lets us simply use the bigram
            # code for unigram experiments, although unfortunately that
            # preserves the O(nk^2) runtime instead of letting us speed 
            # up to O(nk).
            self.A = A.repeat(self.k, 1)
        else:
            # A is already a full matrix giving p(t | s).
            self.A = A

        WB = self._ThetaB @ self._E.t()  # inner products of tag weights and word embeddings
        # B = F.softmax(WB, dim=1)         # run softmax on those inner products to get emission distributions
        B = WB
        self.B = B.clone()
        self.B[self.eos_t, :] = -inf       # but don't guess: EOS_TAG can't emit any column's word (only EOS_WORD)
        self.B[self.bos_t, :] = -inf       # same for BOS_TAG (although BOS_TAG will already be ruled out by other factors)

    def printAB(self):
        """Print the A and B matrices in a more human-readable format (tab-separated)."""
        print("Transition matrix A:")
        col_headers = [""] + [self.tagset[t] for t in range(self.A.size(1))]
        print("\t".join(col_headers))
        for s in range(self.A.shape[0]):   # rows
            row = [self.tagset[s]] + [f"{self.A[s,t]:.3f}" for t in range(self.A.size(1))]
            print("\t".join(row))
        print("\nEmission matrix B:")        
        col_headers = [""] + [self.vocab[w] for w in range(self.B.size(1))]
        print("\t".join(col_headers))
        for t in range(self.A.size(0)):   # rows
            row = [self.tagset[t]] + [f"{self.B[t,w]:.3f}" for w in range(self.B.size(1))]
            print("\t".join(row))
        print("\n")

    def log_prob(self, sentence: Sentence, corpus: TaggedCorpus) -> Tensor:
        """Compute the conditional log probability of a single sentence under the current
        model parameters.  If the sentence is not fully tagged, the probability
        will marginalize over all possible tags.  

        When the logging level is set to DEBUG, the alpha and beta vectors and posterior counts
        are logged.  You can check this against the ice cream spreadsheet."""
        return self.log_forward(sentence, corpus) - self.log_forward(desupervise(sentence), corpus)
    def pad_with_eos(self, sentence):
        sentence2=sentence[:]
        if len(sentence) < self.max_sen_len:
            for i in range(self.max_sen_len-len(sentence)):
                sentence2.append((sentence[-1][0],sentence[-1][1]))
        return sentence2


    def RNN_update_AB(self, sentence: Sentence, corpus: TaggedCorpus):
        s_len = len(sentence)
        words = [item[0] for item in sentence]
        # get h and h_prime
        h = torch.zeros((self.max_sen_len+1, self.M.shape[0], 1)) # each h[j] is for sigmoid of an index j
        h_prime = torch.zeros((self.max_sen_len+1, self.M.shape[0], 1))
        # h_-1 and h'_{n+1} are both zero-tensors, since no info before BOS and after EOS
        for i in range(1, s_len):
            concat1 = torch.cat((torch.tensor([[1]]), h[i-1], self._E2[self.vocab.index(sentence[i][0])].t().reshape(-1,1)), dim=0)
            concat2 = torch.cat((torch.tensor([[1]]), self._E2[self.vocab.index(sentence[i][0])].t().reshape(-1,1), h[-i]), dim=0)
            h[i] = torch.sigmoid(torch.matmul(self.M, concat1))
            h_prime[-(i+1)] = torch.sigmoid(torch.matmul(self.M_prime, concat2))

        # get Fa and Fb, index seq: j, s, t
        Fa = torch.empty((self.max_sen_len, self.k, self.k, self.f_dim)) # TODO: double check dim: s,t each has n-1 choice
        Fb = torch.empty((self.max_sen_len, self.k, self.V, self.f_dim))
        # update Fa
        Fa[:,:,:,:] += torch.sigmoid(self.UA[:, 0]) # add each of UA's first col to each layer k of Fa
        h1 = h[:s_len,:,:][:,None,None,:,:].expand(-1,self.k,self.k,-1,-1)
        Fa[:,:,:,:] += torch.sigmoid(torch.matmul(torch.squeeze(h1, 4), self.UA[:, 1:1+self.h_dim].t())) # h_{j-2}
        s_eye = torch.matmul(self.eye, self.UA[:, 1+self.h_dim:1+self.h_dim+self.k].t())
        Fa[:,:,:,:] += torch.sigmoid(s_eye[None,:,None,:].expand(s_len,-1,self.k,-1))
        t_eye = torch.matmul(self.eye, self.UA[:, 1+self.h_dim+self.k:1+self.h_dim+self.k*2].t())
        Fa[:,:,:,:] += torch.sigmoid(t_eye[None,None,:,:].expand(s_len,self.k,-1,-1))
        h2 = h_prime[:-1,:,:][:,None,None,:,:].expand(-1,self.k,self.k,-1,-1) # TODO: check dimension/index
        Fa[:,:,:,:] += torch.sigmoid(torch.matmul(torch.squeeze(h2,4), self.UA[:, 1+self.h_dim+self.k*2:].t()))
        
        # update Fb
        Fb[:,:,:,:] += torch.sigmoid(self.UB[:, 0])
        h1 = h[1:,:,:][:,None,None,:,:].expand(-1,self.k,self.V,-1,-1) # TODO: check dimension/index
        Fb[:,:,:,:] =  torch.sigmoid(torch.matmul(torch.squeeze(h1, 4), self.UB[:, 1:1+self.h_dim].t())) # h_{j-1}
        t_eye = torch.matmul(self.eye, self.UB[:, 1+self.h_dim:1+self.h_dim+self.k].t())
        Fb[:,:,:,:] = torch.sigmoid(t_eye[None,:,None,:].expand(s_len,-1,self.V,-1))
        w_eye = torch.matmul(self._E, self.UB[:, 1+self.h_dim+self.k:1+self.h_dim+self.k+self.d].t()) # Vxd x dxf_dim
        Fb[:,:,:,:] = torch.sigmoid(w_eye[None,None,:,:].expand(s_len,self.k,-1,-1))
        h2 = h_prime[:-1,:,:][:,None,None,:,:].expand(-1,self.k,self.V,-1,-1) # TODO: check dimension/index
        Fb[:,:,:,:] =  torch.sigmoid(torch.matmul(torch.squeeze(h2,4), self.UB[:, 1+self.h_dim+self.k+self.d:].t()))

        # update phi_A phi_B
        self.A = torch.squeeze(torch.matmul(Fa, self.ThetaA), 3) # size is s_len, self.k, self.k
        self.B = torch.squeeze(torch.matmul(Fb, self.ThetaB), 3)

    def log_forward(self, sentence: Sentence, corpus: TaggedCorpus) -> Tensor:
        """Run the forward algorithm from the handout on a tagged, untagged, 
        or partially tagged sentence.  Return log Z (the log of the forward 
        probability).

        The corpus from which this sentence was drawn is also passed in as an
        argument, to help with integerization and check that we're 
        integerizing correctly."""
        orig_len = len(sentence)
        #pdb.set_trace()
        sentence2 = self.pad_with_eos(sentence)
        sent = self._integerize_sentence(sentence2, corpus)

        # The "nice" way to construct alpha is by appending to a List[Tensor] at each
        # step.  But to better match the notation in the handout, we'll instead preallocate
        # a list of length n+2 so that we can assign directly to alpha[j].
        alpha = [-float("Inf")*torch.ones(self.k) for _ in sent] # very small values close to 0
        alpha[0][self.bos_t] = 0  # handling the first 
        #print("in log forward", self.A.shape, self.B.shape)
       # pdb.set_trace()
        for j in range(1,orig_len-1):
            wi, ti = sent[j]
            if self.birnn:
                # for now, we simply assume either tags are all present or all absent
                if ti == None:
                    x = alpha[j-1].reshape(-1,1) + self.A[j-1]
                    alpha[j] = logsumexp_new(x + self.B[j,:,wi].reshape(1,-1), dim=0, keepdim=False, safe_inf=True)
                else:
                    x = alpha[j-1].reshape(-1) + self.A[j-1,:,ti]
                    #(j, len(sentence), self.B.shape)
                    if wi == 18459:
                        pdb.set_trace()
                    alpha[j][ti] = logsumexp_new(x + self.B[j,ti,wi], dim=0, keepdim=False, safe_inf=True)
            else: 
                if ti == None:
                    x = alpha[j-1].reshape(-1,1) + self.A # we put self.A into log space as well so we dont take log here
                    alpha[j] = logsumexp_new(x + self.B[:,wi].reshape(1,-1), dim=0, keepdim=False, safe_inf=True) # same. B is in log space
                else:
                    x = alpha[j-1] + self.A[:,ti]
                    alpha[j][ti] = logsumexp_new(x + self.B[ti,wi], dim=0, keepdim=False, safe_inf=True)

        # handeling the last tag 
        if self.birnn:
             alpha[-1][self.eos_t]= logsumexp_new(alpha[-2]+ self.A[-1,:,self.eos_t],dim=0, keepdim=False, safe_inf=True)
        else:
            alpha[-1][self.eos_t]= logsumexp_new(alpha[-2]+ self.A[:,self.eos_t],dim=0, keepdim=False, safe_inf=True)
        return alpha[-1][self.eos_t] # Z

    def viterbi_tagging(self, sentence: Sentence, corpus: TaggedCorpus) -> Sentence:
        """Find the most probable tagging for the given sentence, according to the
        current model."""

        # Note: This code is mainly copied from the forward algorithm.
        # We just switch to using max, and follow backpointers.
        # I've continued to call the vector alpha rather than mu.

        sent = self._integerize_sentence(sentence, corpus)

        mu = [-float("Inf")*torch.ones(self.k) for _ in sent] # very small values close to 0
        backpointer=[torch.empty(self.k) for _ in sent]
        mu[0][self.bos_t]=0.  # handling the first 
        for j in range(1,len(sentence)-1):
            wi, ti = sent[j]
            x = mu[j-1].reshape(-1,1) + self.A
            max_mat = torch.max(x + self.B[:,wi].reshape(1,-1),0) 
            mu[j] = max_mat[0]   # alpha values
            backpointer[j] = max_mat[1]
        # handeling the last tag
        max_mat = torch.max(mu[-2].reshape(-1,1)+ self.A, 0)
        mu[-1] = max_mat[0]
        backpointer[-1] = max_mat[1]
        prev_t = self.eos_t
        seq = []
        for i in range(len(sentence)-1,-1,-1):
            word = self.vocab[sent[i][0]]
            tag = self.tagset[prev_t]
           
            prev_t = backpointer[i][prev_t]
            seq.append((word,tag))
        seq.reverse()
        return list(seq)
            

    def train(self, 
              corpus: TaggedCorpus,
              loss: Callable[[CRFModel], float],
              tolerance=0.001,
              minibatch_size: int = 1, evalbatch_size: int = 500,
              lr: float = 1.0, reg: float = 0.0, 
              save_path: Path = Path("my_crf.pkl")) -> None:
        """Train the CRF on the given training corpus, starting at the current parameters.
        The minibatch size controls how often we do an update.
        (Recommended to be larger than 1 for speed; can be inf for the whole training corpus.)
        The evalbatch size controls how often we evaluate (e.g., on a development corpus).
        We will stop when evaluation loss is not better than the last evalbatch by at least the
        tolerance; in particular, we will stop if we evaluation loss is getting worse (overfitting).
        lr is the learning rate, and reg is an L2 batch regularization coefficient."""

        # This is relatively generic training code.  Notice however that the
        # updateAB step before each minibatch produces A, B matrices that are
        # then shared by all sentences in the minibatch.

        # All of the sentences in a minibatch could be treated in parallel,
        # since they use the same parameters.  The code below treats them
        # in series, but if you were using a GPU, you could get speedups
        # by writing the forward algorithm using higher-dimensional tensor 
        # operations that update alpha[j-1] to alpha[j] for all the sentences
        # in the minibatch at once, and then PyTorch could actually take
        # better advantage of hardware parallelism.

        assert minibatch_size > 0
        if minibatch_size > len(corpus):
            minibatch_size = len(corpus)  # no point in having a minibatch larger than the corpus
        assert reg >= 0

        old_dev_loss: Optional[float] = None    # we'll keep track of the dev loss here

        optimizer = torch.optim.SGD(self.parameters(), lr=lr)  # optimizer knows what the params are
        if self.birnn:
            self.A = None
            self.B = None
            #self.RNN_update_AB(sentence, corpus)
            #print('A B initialized', self.A.shape, self.B.shape)
        else:
            self.updateAB()                                    # compute A and B matrices from current params
        log_likelihood = tensor(0.0, device=self.device)       # accumulator for minibatch log_likelihood
        for m, sentence in tqdm(enumerate(corpus.draw_sentences_forever())):
            # Before we process the new sentence, we'll take stock of the preceding
            # examples.  (It would feel more natural to do this at the end of each
            # iteration instead of the start of the next one.  However, we'd also like
            # to do it at the start of the first time through the loop, to print out
            # the dev loss on the initial parameters before the first example.)

            # m is the number of examples we've seen so far.
            # If we're at the end of a minibatch, do an update.
            print("sentence")
            sentence2 = self.pad_with_eos(sentence)
            if self.birnn:
                s_len = len(sentence)
                # self.A = torch.rand((s_len+1, self.k, self.k))
                # self.B = torch.rand((s_len, self.k, self.V))
               # pdb.set_trace()
                #self.B = torch.rand((self.V, self.k))
               # pdb.set_trace()
                self.RNN_update_AB(sentence2, corpus)
            if m % minibatch_size == 0 and m > 0:
                pdb.set_trace()
            #if (not self.birnn and m % minibatch_size == 0 and m > 0) or (self.birnn):
                #input(f"Training log-likelihood per example: {log_likelihood.item()/minibatch_size:.3f} nats")
                logging.debug(f"Training log-likelihood per example: {log_likelihood.item()/minibatch_size:.3f} nats")
                optimizer.zero_grad()          # backward pass will add to existing gradient, so zero it
                objective = -log_likelihood + (minibatch_size/corpus.num_tokens()) * reg * self.params_L2()
                objective.backward()           # compute gradient of regularized negative log-likelihod
                length = sqrt(sum((x.grad*x.grad).sum().item() for x in self.parameters()))
                logging.debug(f"Size of gradient vector: {length}")  # should approach 0 for large minibatch at local min
                optimizer.step()               # SGD step
                if self.birnn:
                    pdb.set_trace()
                    self.RNN_update_AB(sentence, corpus)
                else:
                    self.updateAB()                # update A and B matrices from new params
                # self.printAB()
                log_likelihood = tensor(0.0, device=self.device)    # reset accumulator for next minibatch

            # If we're at the end of an eval batch, or at the start of training, evaluate.
            if m % evalbatch_size == 0:
                with torch.no_grad():       # don't retain gradients during evaluation
                    dev_loss = loss(self)   # this will print its own log messages
                    
                if old_dev_loss is not None and dev_loss >= old_dev_loss * (1-tolerance):
               # if old_dev_loss is not None and dev_loss >= old_dev_loss:
                    # we haven't gotten much better, so stop
                    self.save(save_path)  # Store this model, in case we'd like to restore it later.
                    break
                old_dev_loss = dev_loss            # remember for next eval batch
                
            # Finally, add likelihood of sentence m to the minibatch objective.
            log_likelihood = log_likelihood + self.log_prob(sentence, corpus)


    def save(self, destination: Path) -> None:
        import pickle
        with open(destination, mode="wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info(f"Saved model to {destination}")


    @classmethod
    def load(cls, source: Path) -> CRFModel:
        import pickle  # for loading/saving Python objects
        logging.info(f"Loading model from {source}")
        with open(source, mode="rb") as f:
            result = pickle.load(f)
            logging.info(f"Loaded model from {source}")
            return result
