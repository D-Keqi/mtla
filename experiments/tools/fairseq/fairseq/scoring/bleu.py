# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2025 Keqi Deng (University of Cambridge)
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import ctypes
import evaluate
import math
import re
import string
import sys
from dataclasses import dataclass, field

import torch
from fairseq.dataclass import FairseqDataclass
from fairseq.scoring import BaseScorer, register_scorer
from fairseq.scoring.tokenizer import EvaluationTokenizer


class BleuStat(ctypes.Structure):
    _fields_ = [
        ("reflen", ctypes.c_size_t),
        ("predlen", ctypes.c_size_t),
        ("match1", ctypes.c_size_t),
        ("count1", ctypes.c_size_t),
        ("match2", ctypes.c_size_t),
        ("count2", ctypes.c_size_t),
        ("match3", ctypes.c_size_t),
        ("count3", ctypes.c_size_t),
        ("match4", ctypes.c_size_t),
        ("count4", ctypes.c_size_t),
    ]


@dataclass
class SacrebleuConfig(FairseqDataclass):
    sacrebleu_tokenizer: EvaluationTokenizer.ALL_TOKENIZER_TYPES = field(
        default="13a", metadata={"help": "tokenizer"}
    )
    sacrebleu_lowercase: bool = field(
        default=False, metadata={"help": "apply lowercasing"}
    )
    sacrebleu_char_level: bool = field(
        default=False, metadata={"help": "evaluate at character level"}
    )


@register_scorer("sacrebleu", dataclass=SacrebleuConfig)
class SacrebleuScorer(BaseScorer):
    def __init__(self, cfg):
        super(SacrebleuScorer, self).__init__(cfg)
        import sacrebleu

        self.sacrebleu = sacrebleu
        self.tokenizer = EvaluationTokenizer(
            tokenizer_type=cfg.sacrebleu_tokenizer,
            lowercase=cfg.sacrebleu_lowercase,
            character_tokenization=cfg.sacrebleu_char_level,
        )

    def add_string(self, ref, pred):
        self.ref.append(self.tokenizer.tokenize(ref))
        self.pred.append(self.tokenizer.tokenize(pred))

    def _score(self, order=4):
        if order != 4:
            raise NotImplementedError
        # tokenization and lowercasing are performed by self.tokenizer instead.
        return self.sacrebleu.corpus_bleu(self.pred, [self.ref], tokenize="none")

    def score(self, order=4):
        return self._score(order).score

    def result_string(self, order=4):
        return self._score(order).format()


@dataclass
class BleuConfig(FairseqDataclass):
    pad: int = field(default=1, metadata={"help": "padding index"})
    eos: int = field(default=2, metadata={"help": "eos index"})
    unk: int = field(default=3, metadata={"help": "unk index"})


@register_scorer("bleu", dataclass=BleuConfig)
class Scorer(object):
    def __init__(self, cfg):
        self.stat = BleuStat()
        self.pad = cfg.pad
        self.eos = cfg.eos
        self.unk = cfg.unk

        try:
            from fairseq import libbleu
        except ImportError as e:
            sys.stderr.write(
                "ERROR: missing libbleu.so. run `pip install --editable .`\n"
            )
            raise e

        self.C = ctypes.cdll.LoadLibrary(libbleu.__file__)

        self.reset()

    def reset(self, one_init=False):
        if one_init:
            self.C.bleu_one_init(ctypes.byref(self.stat))
        else:
            self.C.bleu_zero_init(ctypes.byref(self.stat))

    def add(self, ref, pred):
        if not isinstance(ref, torch.IntTensor):
            raise TypeError("ref must be a torch.IntTensor (got {})".format(type(ref)))
        if not isinstance(pred, torch.IntTensor):
            raise TypeError("pred must be a torch.IntTensor(got {})".format(type(pred)))

        # don't match unknown words
        rref = ref.clone()
        assert not rref.lt(0).any()
        rref[rref.eq(self.unk)] = -999

        rref = rref.contiguous().view(-1)
        pred = pred.contiguous().view(-1)

        self.C.bleu_add(
            ctypes.byref(self.stat),
            ctypes.c_size_t(rref.size(0)),
            ctypes.c_void_p(rref.data_ptr()),
            ctypes.c_size_t(pred.size(0)),
            ctypes.c_void_p(pred.data_ptr()),
            ctypes.c_int(self.pad),
            ctypes.c_int(self.eos),
        )

    def score(self, order=4):
        psum = sum(
            math.log(p) if p > 0 else float("-Inf") for p in self.precision()[:order]
        )
        return self.brevity() * math.exp(psum / order) * 100

    def precision(self):
        def ratio(a, b):
            return a / b if b > 0 else 0

        return [
            ratio(self.stat.match1, self.stat.count1),
            ratio(self.stat.match2, self.stat.count2),
            ratio(self.stat.match3, self.stat.count3),
            ratio(self.stat.match4, self.stat.count4),
        ]

    def brevity(self):
        r = self.stat.reflen / self.stat.predlen
        return min(1, math.exp(1 - r))

    def result_string(self, order=4):
        assert order <= 4, "BLEU scores for order > 4 aren't supported"
        fmt = "BLEU{} = {:2.2f}, {:2.1f}"
        for _ in range(1, order):
            fmt += "/{:2.1f}"
        fmt += " (BP={:.3f}, ratio={:.3f}, syslen={}, reflen={})"
        bleup = [p * 100 for p in self.precision()[:order]]
        return fmt.format(
            order,
            self.score(order=order),
            *bleup,
            self.brevity(),
            self.stat.predlen / self.stat.reflen,
            self.stat.predlen,
            self.stat.reflen,
        )


@dataclass
class RougeLConfig(FairseqDataclass):
    rouge_lowercase: bool = field(
        default=True, metadata={"help": "apply lowercasing before evaluation"}
    )
    rouge_stemming: bool = field(
        default=True, metadata={"help": "apply Porter stemmer before evaluation"}
    )
    rouge_corpus_level: bool = field(
        default=True,
        metadata={
            "help": "compute corpus-level ROUGE-L score instead of sentence-level"
        },
    )
    rouge_detokenize: bool = field(
        default=True, metadata={"help": "apply detokenization before evaluation"}
    )


@register_scorer("rouge_l", dataclass=RougeLConfig)
class RougeLScorer(BaseScorer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.stemmer = self._build_stemmer() if cfg.rouge_stemming else None
        self.detok = self._build_detokenizer() if cfg.rouge_detokenize else None
        self.rouge = evaluate.load("rouge")
        self.refs = []
        self.preds = []

    def _build_stemmer(self):
        try:
            from nltk.stem import PorterStemmer

            return PorterStemmer()
        except ImportError:
            raise ImportError("Please install NLTK: `pip install nltk`")

    def _build_detokenizer(self):
        try:
            from sacremoses import MosesDetokenizer

            return MosesDetokenizer()
        except ImportError:
            raise ImportError("Please install sacremoses: `pip install sacremoses`")

    def _detokenize(self, text):
        if self.detok:
            words = text.strip().split()
            return self.detok.detokenize(words)
        return text

    def _preprocess(self, text):
        text = self._detokenize(text)
        if self.cfg.rouge_lowercase:
            text = text.lower()

        # Remove punctuation
        text = re.sub(rf"[{re.escape(string.punctuation)}]", "", text)

        if self.cfg.rouge_stemming and self.stemmer:
            words = text.strip().split()
            text = " ".join([self.stemmer.stem(word) for word in words])
        return text

    def add_string(self, ref, pred):
        self.refs.append(self._preprocess(ref))
        self.preds.append(self._preprocess(pred))

    def _score_corpus_level(self):
        results = self.rouge.compute(predictions=self.preds, references=self.refs)
        return {
            "rouge1": results["rouge1"] * 100,
            "rouge2": results["rouge2"] * 100,
            "rougeL": results["rougeL"] * 100,
        }

    def _score_sentence_level(self):
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        for ref, pred in zip(self.refs, self.preds):
            result = self.rouge.compute(predictions=[pred], references=[ref])
            rouge1_scores.append(result["rouge1"])
            rouge2_scores.append(result["rouge2"])
            rougeL_scores.append(result["rougeL"])
        return {
            "rouge1": sum(rouge1_scores) / len(rouge1_scores) * 100,
            "rouge2": sum(rouge2_scores) / len(rouge2_scores) * 100,
            "rougeL": sum(rougeL_scores) / len(rougeL_scores) * 100,
        }

    def score(self):
        if self.cfg.rouge_corpus_level:
            return self._score_corpus_level()
        else:
            return self._score_sentence_level()

    def result_string(self):
        scores = self.score()
        level = "corpus-level" if self.cfg.rouge_corpus_level else "sentence-level"
        return (
            f"ROUGE-1 ({level}): {scores['rouge1']:.2f}\n"
            f"ROUGE-2 ({level}): {scores['rouge2']:.2f}\n"
            f"ROUGE-L ({level}): {scores['rougeL']:.2f}"
        )
