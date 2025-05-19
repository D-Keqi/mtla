# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from fairseq import utils
from fairseq.logging import metrics
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    LabelSmoothedCrossEntropyCriterionConfig,
)
from fairseq.data.data_utils import lengths_to_mask


@dataclass
class LabelSmoothedCrossEntropyWithCtcCriterionConfig(
    LabelSmoothedCrossEntropyCriterionConfig
):
    ctc_weight: float = field(default=1.0, metadata={"help": "weight for CTC loss"})


@register_criterion(
    "label_smoothed_cross_entropy_with_ctc",
    dataclass=LabelSmoothedCrossEntropyWithCtcCriterionConfig,
)
class LabelSmoothedCrossEntropyWithCtcCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size,
        report_accuracy,
        ctc_weight,
    ):
        super().__init__(
            task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy
        )
        self.ctc_weight = ctc_weight

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

        ctc_loss = torch.tensor(0.0).type_as(loss)
        if self.ctc_weight > 0.0:
            ctc_lprobs, ctc_lens = model.get_ctc_output(net_output, sample)
            ctc_tgt, ctc_tgt_lens = model.get_ctc_target(sample)
            ctc_tgt_mask = lengths_to_mask(ctc_tgt_lens)
            ctc_tgt_flat = ctc_tgt.masked_select(ctc_tgt_mask)
            reduction = "sum" if reduce else "none"
            ctc_loss = (
                F.ctc_loss(
                    ctc_lprobs,
                    ctc_tgt_flat,
                    ctc_lens,
                    ctc_tgt_lens,
                    reduction=reduction,
                    zero_infinity=True,
                )
                * self.ctc_weight
            )
        loss += ctc_loss

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": utils.item(loss.data),
            "nll_loss": utils.item(nll_loss.data),
            "ctc_loss": utils.item(ctc_loss.data),
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        super().reduce_metrics(logging_outputs)
        loss_sum = sum(log.get("ctc_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "ctc_loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion(
    "label_smoothed_cross_entropy_with_ctc_ic",
    dataclass=LabelSmoothedCrossEntropyWithCtcCriterionConfig,
)
class LabelSmoothedCrossEntropyWithCtcCriterionIC(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size,
        report_accuracy,
        ctc_weight,
    ):
        super().__init__(
            task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy
        )
        self.ctc_weight = ctc_weight

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

        ctc_loss = torch.tensor(0.0).type_as(loss)
        if self.ctc_weight > 0.0:
            ctc_lprobs, ctc_lens = model.get_ctc_output(net_output, sample)
            ctc_tgt, ctc_tgt_lens = model.get_ctc_target(sample)
            ctc_tgt = ctc_tgt[:, 1:]
            ctc_tgt_lens = ctc_tgt_lens - 1
            ctc_tgt_mask = lengths_to_mask(ctc_tgt_lens)
            ctc_tgt_flat = ctc_tgt.masked_select(ctc_tgt_mask)
            reduction = "sum" if reduce else "none"
            ctc_loss = (
                F.ctc_loss(
                    ctc_lprobs,
                    ctc_tgt_flat,
                    ctc_lens,
                    ctc_tgt_lens,
                    reduction=reduction,
                    zero_infinity=True,
                )
                * self.ctc_weight
            )
        loss += ctc_loss

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": utils.item(loss.data),
            "nll_loss": utils.item(nll_loss.data),
            "ctc_loss": utils.item(ctc_loss.data),
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        super().reduce_metrics(logging_outputs)
        loss_sum = sum(log.get("ctc_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "ctc_loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs_ic = lprobs[:, :1]
        lprobs_left = lprobs[:, 1:]
        target = model.get_targets(sample, net_output)
        target_ic = target[:, :1]
        target_left = target[:, 1:]
        return (
            lprobs_ic.reshape(-1, lprobs_ic.size(-1)),
            target_ic.reshape(-1),
            lprobs_left.reshape(-1, lprobs_left.size(-1)),
            target_left.reshape(-1),
        )

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs_ic, target_ic, lprobs_left, target_left = self.get_lprobs_and_target(
            model, net_output, sample
        )
        loss_ic, nll_loss_ic = label_smoothed_nll_loss(
            lprobs_ic,
            target_ic,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        loss_left, nll_loss_left = label_smoothed_nll_loss(
            lprobs_left,
            target_left,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss_ic * 5 + loss_left, nll_loss_ic * 5 + nll_loss_left

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target, _, _ = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total
