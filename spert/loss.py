from abc import ABC
import torch.nn.functional as F
import torch
from torch import nn as nn
from spert import util
from spert.memory import ContrastMemoryNo

eps = 1e-7

# def sample_pos_negK(entity_features_s, entity_features_TeaE, k):
#     entity_features_s_sample = torch.zeros([entity_features_TeaE.shape[0], k + 1, entity_features_TeaE.shape[1]]).to(
#         entity_features_TeaE.device)
#     for i in range(entity_features_TeaE.shape[0]):
#         entity_features_s_sample[i, 0, :] = entity_features_s[i]
#         k_i = 0
#         for j in range(entity_features_s.shape[0]):
#             if k_i >= k:
#                 break
#             if j != i:
#                 entity_features_s_sample[i, 1 + k_i, :] = entity_features_s[j]
#                 k_i += 1
#     entity_out_s_TeaE = torch.bmm(entity_features_TeaE.unsqueeze(1), entity_features_s_sample.permute(0, 2, 1)).squeeze(1)  # [bsz, 1, dim], [dsz, dim, k+1]
#     return entity_out_s_TeaE

class COSNCELoss(nn.Module):
    def __init__(self):
        super(COSNCELoss, self).__init__()

    def g(self, x, y):
        return 1 - (x*y).sum(-1)

    def forward(self, r_s, r_t):
        '''
        :param r_t: [b, seq, hid]
        :param r_s: [b, seq, hid]
        :return: loss value
        '''
        r_t = F.normalize(r_t, p=2, dim=-1)
        r_s = F.normalize(r_s, p=2, dim=-1)

        neg_samples = []

        batch_size = r_t.size(0)
        hid_size = r_t.size(2)
        K = batch_size - 1
        for i in range(batch_size):
            neg_samples.append([])
            for j in range(batch_size):
                if j==i:
                    continue
                neg_samples[i].append(r_t[j]) # [seq, hid]

        for i in range(batch_size):
            neg_samples[i] = torch.stack(neg_samples[i], 1) # [seq, K, hid]

        neg_samples = torch.cat(neg_samples, 0) # [b*seq, K, hid]

        r_t = r_t.view(-1, 1, hid_size) # [b*seq, 1, hid]
        r_s = r_s.view(-1, 1, hid_size) # [b*seq, 1, hid]

        pos_score = self.g(r_s, r_t) # [b*seq, 1]
        neg_scores = self.g(r_s, neg_samples) # [b*seq, K]
        neg_scores = 2 - (neg_scores - pos_score) # [b*seq, K]
        neg_score = neg_scores.sum(-1, keepdim=True)/(2*K) # [b*seq, 1]

        loss = pos_score + neg_score

        loss = loss.sum()/(batch_size*200*(K+1)*4)

        return loss


def sample_pos_negK(entity_features_s, entity_features_TeaE, k):
    entity_features_s_sample = []
    batch_size = entity_features_TeaE.shape[0]  # max k = batch_size - 1
    for i in range(batch_size):
        k_i = 0
        entity_features_s_sample.append([])
        for j in range(batch_size):
            if k_i >= k:
                break
            if j == i:
                continue
            else:
                entity_features_s_sample[i].append(entity_features_s[j]) # [seq, hid]
                k_i += 1
    for i in range(batch_size):
        entity_features_s_sample[i] = torch.stack(entity_features_s_sample[i], 1)  # [seq, K+1, hid]
    entity_features_s_sample = torch.cat(entity_features_s_sample, 0)  # [b*seq, K+1, hid]
    entity_out_s_TeaE = torch.bmm(entity_features_TeaE.unsqueeze(1), entity_features_s_sample.permute(0, 2, 1)).squeeze(1)  # [bsz, 1, dim], [dsz, dim, k+1]
    return entity_out_s_TeaE


# def sample_pos_negK(entity_features_s, entity_features_TeaE, k):
#     entity_features_s_sample = torch.zeros([entity_features_TeaE.shape[0], k + 1, entity_features_TeaE.shape[1]]).to(
#         entity_features_TeaE.device)
#     entity_features_s_sample[:, 0, :] = entity_features_s  # all pos
#     for i in range(entity_features_TeaE.shape[0]):
#         k_i = 0
#         for j in range(entity_features_s.shape[0]):
#             if k_i >= k:
#                 break
#             if j != i:
#                 entity_features_s_sample[i, 1 + k_i, :] = entity_features_s[j]
#                 k_i += 1
#     entity_out_s_TeaE = torch.bmm(entity_features_TeaE.unsqueeze(1), entity_features_s_sample.permute(0, 2, 1)).squeeze(1)  # [bsz, 1, dim], [dsz, dim, k+1]
#     return entity_out_s_TeaE

class Loss(ABC):
    def compute(self, *args, **kwargs):
        pass

class H(nn.Module):
    def __init__(self, hidden_size, M, N, t):
        super(H, self).__init__()
        self.t_transform = nn.Linear(hidden_size, hidden_size)
        self.s_transform = nn.Linear(hidden_size, hidden_size)

        self.M = M
        self.N = N

        self.t = t

    def forward(self, t_rep, s_rep): # [b, hidden] [b, hidden]
        t_rep = self.t_transform(t_rep)
        s_rep = self.s_transforms(s_rep)

        t_rep = F.normalize(t_rep, p=2, dim=-1)
        s_rep = F.normalize(s_rep, p=2, dim=-1)

        score = (t_rep*s_rep).sum(dim=-1, keepdim=True) # [b, 1]
        score = score/self.t

        score = torch.exp(score)/(torch.exp(score) + self.N/self.M)  # [b, 1]

        return score # [b, 1]


class SpERTLoss(Loss):
    def __init__(self, rel_criterion, entity_criterion, model, optimizer, scheduler, max_grad_norm):
        self._rel_criterion = rel_criterion
        self._entity_criterion = entity_criterion
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._max_grad_norm = max_grad_norm

    def compute(self, entity_logits, rel_logits, entity_types, rel_types, entity_sample_masks, rel_sample_masks):
        # entity loss
        entity_logits = entity_logits.view(-1, entity_logits.shape[-1])
        entity_types = entity_types.view(-1)
        entity_sample_masks = entity_sample_masks.view(-1).float()

        entity_loss = self._entity_criterion(entity_logits, entity_types)
        entity_loss = (entity_loss * entity_sample_masks).sum() / entity_sample_masks.sum()

        # relation loss
        rel_sample_masks = rel_sample_masks.view(-1).float()
        rel_count = rel_sample_masks.sum()

        if rel_count.item() != 0:
            rel_logits = rel_logits.view(-1, rel_logits.shape[-1])
            rel_types = rel_types.view(-1, rel_types.shape[-1])

            rel_loss = self._rel_criterion(rel_logits, rel_types)
            rel_loss = rel_loss.sum(-1) / rel_loss.shape[-1]
            rel_loss = (rel_loss * rel_sample_masks).sum() / rel_count

            # joint loss
            train_loss = entity_loss + rel_loss
        else:
            # corner case: no positive/negative relation samples
            train_loss = entity_loss

        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
        self._optimizer.step()
        self._scheduler.step()
        self._model.zero_grad()
        return train_loss.item()

    def contrastive_compute(self, entity_logits, rel_logits, contrastive_entity_loss, contrastive_rel_loss, entity_types, rel_types, entity_sample_masks, rel_sample_masks):
        # entity loss
        entity_logits = entity_logits.view(-1, entity_logits.shape[-1])
        entity_types = entity_types.view(-1)
        entity_sample_masks = entity_sample_masks.view(-1).float()

        entity_loss = self._entity_criterion(entity_logits, entity_types)  #
        entity_loss = (entity_loss * entity_sample_masks).sum() / entity_sample_masks.sum()

        # relation loss
        rel_sample_masks = rel_sample_masks.view(-1).float()
        rel_count = rel_sample_masks.sum()

        if rel_count.item() != 0:
            rel_logits = rel_logits.view(-1, rel_logits.shape[-1])
            rel_types = rel_types.view(-1, rel_types.shape[-1])

            rel_loss = self._rel_criterion(rel_logits, rel_types)
            rel_loss = rel_loss.sum(-1) / rel_loss.shape[-1]
            rel_loss = (rel_loss * rel_sample_masks).sum() / rel_count

            # joint loss
            train_loss = entity_loss + rel_loss + contrastive_entity_loss + contrastive_rel_loss
        else:
            # corner case: no positive/negative relation samples
            train_loss = entity_loss + contrastive_entity_loss

        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
        self._optimizer.step()
        self._scheduler.step()
        self._model.zero_grad()
        return train_loss.item()


class CRDLoss(nn.Module):
    """CRD Loss function
    includes two symmetric parts:
    (a) using teacher as anchor, choose positive and negatives over the student side
    (b) using student as anchor, choose positive and negatives over the teacher side

    Args:
        opt.s_dim: the dimension of student's feature (feat_dim_in)
        opt.t_dim: the dimension of teacher's feature (feat_dim_in)
        opt.feat_dim: the dimension of the projection space  (feat_dim_out)
        opt.nce_k: number of negatives paired with each positive
        opt.nce_t: the temperature
        opt.nce_m: the momentum for updating the memory buffer
        opt.n_data: the number of samples in the training set, therefor the memory buffer is: opt.n_data x opt.feat_dim
    """
    def __init__(self, n_data, entity_dim_in=1024, rel_dim_in=1024, dim_out=128, crd_k=10, crd_t=0.07, crd_m=0.5):
        # batch_size
        # t, temperature
        super(CRDLoss, self).__init__()
        self.entity_embed_s = Embed(entity_dim_in, dim_out)
        self.entity_embed_TeaE = Embed(entity_dim_in, dim_out)
        self.entity_embed_TeaR = Embed(entity_dim_in, dim_out)
        self.rel_embed_s = Embed(rel_dim_in, dim_out)
        self.rel_embed_TeaE = Embed(rel_dim_in, dim_out)
        self.rel_embed_TeaR = Embed(rel_dim_in, dim_out)
        self.k = crd_k
        # remember add
        self.contrast = ContrastMemoryNo(dim_out, n_data, self.k, crd_t, crd_m)
        # self.embed_t = Embed(opt.t_dim, opt.feat_dim)
        # self.contrast = ContrastMemory(opt.feat_dim, opt.n_data, opt.nce_k, opt.nce_t, opt.nce_m)
        self.criterion_s_entity = ContrastLoss()
        self.criterion_s_rel = ContrastLoss()

    def forward(self, entity_features_s, rel_features_s, entity_features_TeaE, rel_features_TeaE, entity_logits_TeaE, rel_logits_TeaE, entity_features_TeaR, rel_features_TeaR, entity_logits_TeaR, rel_logits_TeaR, contrast_idx, idx, M):
        """
        Args:
            f_s: the feature of student network, size [batch_size, s_dim]
            f_t: the feature of teacher network, size [batch_size, t_dim]

        Returns:
            The contrastive loss
        """
        # wt
        entity_TeaE_wt = torch.max(torch.softmax(entity_logits_TeaE, dim=2), dim=2)[1].data.view(-1)  
        rel_TeaE_wt = torch.max(torch.softmax(rel_logits_TeaE, dim=2), dim=2)[1].data.view(-1)
        entity_TeaR_wt = torch.max(torch.softmax(entity_logits_TeaR, dim=2), dim=2)[1].data.view(-1)  
        rel_TeaR_wt = torch.max(torch.softmax(rel_logits_TeaR, dim=2), dim=2)[1].data.view(-1)

        # entity
        # util.to_device(entity_features_s, self._device)
        entity_features_s = self.entity_embed_s(entity_features_s.view(-1, entity_features_s.shape[2]))
        entity_features_TeaE = self.entity_embed_TeaE(entity_features_TeaE.view(-1, entity_features_TeaE.shape[2]))
        entity_features_TeaR = self.entity_embed_TeaR(entity_features_TeaR.view(-1, entity_features_TeaR.shape[2]))
        # 构造基于pair的pos、neg数据集
        # remember add contrast_idx, idx ...
        # out_s, out_t = self.contrast(f_s, f_t, idx, contrast_idx)  # so implement it later...
        # entity_out_s_TeaE, entity_out_TeaE = self.contrast(entity_features_s, entity_features_TeaE, idx, contrast_idx)
        # entity_out_s_TeaR, entity_out_TeaR = self.contrast(entity_features_s, entity_features_TeaR, idx, contrast_idx)
        # later explore ...

        entity_out_s_TeaE = sample_pos_negK(entity_features_s, entity_features_TeaE, self.k)
        entity_out_s_TeaE = self.contrast(entity_out_s_TeaE)
        entity_out_s_TeaR = sample_pos_negK(entity_features_s, entity_features_TeaR, self.k)
        entity_out_s_TeaR = self.contrast(entity_out_s_TeaR)
        #
        entity_s_loss = self.criterion_s_entity(entity_out_s_TeaE, entity_TeaE_wt, M) + self.criterion_s_entity(entity_out_s_TeaR, entity_TeaR_wt, M)

        # # rel
        rel_features_s = self.rel_embed_s(rel_features_s.view(-1, rel_features_s.shape[2]))
        rel_features_TeaE = self.rel_embed_TeaE(rel_features_TeaE.view(-1, rel_features_TeaE.shape[2]))
        rel_features_TeaR = self.rel_embed_TeaR(rel_features_TeaR.view(-1, rel_features_TeaR.shape[2]))
        # 构造基于pair的pos、neg数据集
        rel_out_s_TeaE = sample_pos_negK(rel_features_s, rel_features_TeaE, self.k)
        rel_out_s_TeaE = self.contrast(rel_out_s_TeaE)
        rel_out_s_TeaR = sample_pos_negK(rel_features_s, rel_features_TeaR, self.k)
        rel_out_s_TeaR = self.contrast(rel_out_s_TeaR)
        #
        rel_s_loss = self.criterion_s_rel(rel_out_s_TeaE, rel_TeaE_wt, M) + self.criterion_s_rel(rel_out_s_TeaR, rel_TeaR_wt, M)
        return entity_s_loss, rel_s_loss


class ContrastLoss(nn.Module):
    """
    contrastive loss, corresponding to Eq (18)
    n_data: the number of data
    x: the embedding [batch_size, dim]
    """
    def __init__(self):
        super(ContrastLoss, self).__init__()

    def forward(self, x, wt, M):
        bsz = x.shape[0]
        m = x.size(1) - 1  # N

        # noise distribution
        Pn = 1 / float(M)  # Cardinality, M

        # loss for positive pair
        P_pos = x.select(1, 0)  
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()  # C=1
        
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()  # C=0
       

        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz

        return loss


class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = self.linear(x)
        x = self.l2norm(x)
        # x = x.view(x.shape[0], -1)
        return x


class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(-1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

