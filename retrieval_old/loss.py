import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch import Tensor as T

from typing import Tuple, List

class BiEncoderNllLoss(object):
    def calc(
        self,
        vector_1: T,
        vector_2: T,
        positive_idx_per_question: list = None,
        hard_negative_idx_per_question: list = None,
        loss_scale: float = None,
    ) -> Tuple[T, int]:
        """
        Computes nll loss for the given lists of question and ctx vectors.
        Note that although hard_negative_idx_per_question in not currently in use, one can use it for the
        loss modifications. For example - weighted NLL with different factors for hard vs regular negatives.
        :return: a tuple of loss value and amount of correct predictions per batch
        """
        scores = self.get_scores(vector_1, vector_2)

        if len(vector_1.size()) > 1:
            q_num = vector_1.size(0)
            scores = scores.view(q_num, -1)

        softmax_scores = F.log_softmax(scores, dim=1)
        logits = F.softmax(scores, dim=1)

        positive_idx_per_question = list(range(vector_1.size()[0]))

        loss = F.nll_loss(
            softmax_scores,
            torch.tensor(positive_idx_per_question).to(softmax_scores.device),
            reduction="mean",
        )

        max_score, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions_count = (max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)).sum()

        if loss_scale:
            loss.mul_(loss_scale)
        
        return loss, logits, correct_predictions_count

    @staticmethod
    def get_scores(q_vector: T, ctx_vectors: T) -> T:
        f = BiEncoderNllLoss.get_similarity_function()
        return f(q_vector, ctx_vectors)

    @staticmethod
    def get_similarity_function():
        return dot_product_scores

def dot_product_scores(q_vectors: T, ctx_vectors: T) -> T:
    """
    calculates q->ctx scores for every row in ctx_vector
    :param q_vector:
    :param ctx_vector:
    :return:
    """
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    r = torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))
    return r

class BiEncoderNllLos_BAK(object):
    def calc(self, sent1_vectors, sent2_vectors):
        # positive_idx_per_question = list(range(sent1_vectors.size()[0]))
        # hard_negative_idx_per_question = None
        # scores = self.get_scores(sent1_vectors, sent2_vectors)
        
        # if len(sent1_vectors.size()) > 1:
        #     q_num = sent1_vectors.size(0)
        #     scores = scores.view(q_num, -1)

        sent1_vectors = F.normalize(sent1_vectors, dim=-1)
        sent2_vectors = F.normalize(sent2_vectors, dim=-1)

        logits = torch.matmul(sent1_vectors, sent2_vectors.t())
        # print(f"loss.py : {logits}")
        labels = torch.arange(sent1_vectors.shape[0], dtype=torch.long).to(logits.device)
        
        loss_fn = nn.CrossEntropyLoss()
        
        #symmetric loss function
        loss = loss_fn(logits, labels)
        

        # softmax_scores = F.log_softmax(scores, dim=1)
        # loss = F.nll_loss(
        #     softmax_scores,
        #     torch.tensor(positive_idx_per_question).to(softmax_scores.device),
        #     reduction="sum",
        # )


        # max_score, max_idxs = torch.max(softmax_scores, 1)
        # correct_predictions_count = (max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)).sum()

        # softmax_scores = logits
        return loss, logits

    def get_scores(self, sent1_vectors, sent2_vectors):
        r = self.get_similarity_function(sent1_vectors, sent2_vectors)
        return r
    
    # def dot_product_scores(sent1_vectors, sent2_vectors):


    def get_similarity_function(self, sent1_vectors, sent2_vectors):
        """
        calculates q->ctx scores for every row in ctx_vector
        :param q_vector:
        :param ctx_vector:
        :return:
        """
        # q_vector: n1 x D, sent2_vectors: n2 x D, result n1 x n2
        # r = torch.matmul(sent1_vectors, torch.transpose(sent2_vectors, 0, 1))
        
        r = torch.matmul(sent1_vectors, sent2_vectors.t())
        # cos = nn.CosineSimilarity()
        # r = cos(sent1_vectors, sent2_vectors)
        
        return r