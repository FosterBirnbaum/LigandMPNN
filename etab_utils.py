import sys
import time
import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import re

def merge_duplicate_pairE(h_E, E_idx):
    """ Average pair energy tables across bidirectional edges.

    TERMinator edges are represented as two bidirectional edges, and to allow for
    communication between these edges we average the embeddings. In the case for
    pair energies, we transpose the tables to ensure that the pair energy table
    is symmetric upon inverse (e.g. the pair energy between i and j should be
    the same as the pair energy between j and i)

    Args
    ----
    h_E : torch.Tensor
        Pair energies in kNN sparse form
        Shape : n_batch x n_res x k x n_aa x n_aa
    E_idx : torch.LongTensor
        kNN sparse edge indices
        Shape : n_batch x n_res x k

    Returns
    -------
    torch.Tensor
        Pair energies with merged energies for bidirectional edges
        Shape : n_batch x n_res x k x n_aa x n_aa
    """
    try:
        k = E_idx.shape[-1]
        seq_lens = torch.ones(h_E.shape[0]).long().to(h_E.device) * h_E.shape[1]
        if len(h_E.shape) > 4:
            h_E_geometric = h_E.view([-1, 400])
            orig_length = 5
        else:
            h_E_geometric = h_E.view([-1, h_E.shape[-1]])
            orig_length = 4
        split_E_idxs = torch.unbind(E_idx)
        offset = [seq_lens[:i].sum() for i in range(len(seq_lens))]
        split_E_idxs = [e.to(h_E.device) + o for e, o in zip(split_E_idxs, offset)]
        edge_index_row = torch.cat([e.view(-1) for e in split_E_idxs], dim=0)
        edge_index_col = torch.repeat_interleave(torch.arange(edge_index_row.shape[0] // k), k).to(h_E.device)
        edge_index = torch.stack([edge_index_row, edge_index_col])
        merge = merge_duplicate_pairE_geometric(h_E_geometric, edge_index, orig_length)
        merge = merge.view(h_E.shape)
        #old_merge = merge_duplicate_pairE_dense(h_E, E_idx)
        #assert (old_merge == merge).all(), (old_merge, merge)

        return merge
    except RuntimeError as err:
        print(err, file=sys.stderr)
        print("We're handling this error as if it's an out-of-memory error", file=sys.stderr)
        torch.cuda.empty_cache()  # this is probably unnecessary but just in case
        raise err
        return merge_duplicate_pairE_sparse(h_E, E_idx)


def merge_duplicate_pairE_dense(h_E, E_idx):
    """ Dense method to average pair energy tables across bidirectional edges.

    TERMinator edges are represented as two bidirectional edges, and to allow for
    communication between these edges we average the embeddings. In the case for
    pair energies, we transpose the tables to ensure that the pair energy table
    is symmetric upon inverse (e.g. the pair energy between i and j should be
    the same as the pair energy between j and i)

    Args
    ----
    h_E : torch.Tensor
        Pair energies in kNN sparse form
        Shape : n_batch x n_res x k x n_aa x n_aa
    E_idx : torch.LongTensor
        kNN sparse edge indices
        Shape : n_batch x n_res x k

    Returns
    -------
    torch.Tensor
        Pair energies with merged energies for bidirectional edges
        Shape : n_batch x n_res x k x n_aa x n_aa
    """
    dev = h_E.device
    n_batch, n_nodes, _, n_aa, _ = h_E.shape
    # collect edges into NxN tensor shape
    collection = torch.zeros((n_batch, n_nodes, n_nodes, n_aa, n_aa)).to(dev)
    neighbor_idx = E_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, n_aa, n_aa).to(dev)
    collection.scatter_(2, neighbor_idx, h_E)
    # transpose to get same edge in reverse direction
    collection = collection.transpose(1, 2)
    # transpose each pair energy table as well
    collection = collection.transpose(-2, -1)
    # gather reverse edges
    reverse_E = gather_pairEs(collection, E_idx)
    # average h_E and reverse_E at non-zero positions
    merged_E = torch.where(reverse_E != 0, (h_E + reverse_E) / 2, h_E)
    return merged_E


# TODO: rigorous test that this is equiv to the dense version
def merge_duplicate_pairE_sparse(h_E, E_idx):
    """ Sparse method to average pair energy tables across bidirectional edges.

    Note: This method involves a significant slowdown so it's only worth using if memory is an issue.

    TERMinator edges are represented as two bidirectional edges, and to allow for
    communication between these edges we average the embeddings. In the case for
    pair energies, we transpose the tables to ensure that the pair energy table
    is symmetric upon inverse (e.g. the pair energy between i and j should be
    the same as the pair energy between j and i)

    Args
    ----
    h_E : torch.Tensor
        Pair energies in kNN sparse form
        Shape : n_batch x n_res x k x n_aa x n_aa
    E_idx : torch.LongTensor
        kNN sparse edge indices
        Shape : n_batch x n_res x k

    Returns
    -------
    torch.Tensor
        Pair energies with merged energies for bidirectional edges
        Shape : n_batch x n_res x k x n_aa x n_aa
    """
    dev = h_E.device
    n_batch, n_nodes, k, n_aa, _ = h_E.shape
    # convert etab into a sparse etab
    # self idx of the edge
    ref_idx = E_idx[:, :, 0:1].expand(-1, -1, k)
    # sparse idx
    g_idx = torch.cat([E_idx.unsqueeze(1), ref_idx.unsqueeze(1)], dim=1)
    sparse_idx = g_idx.view([n_batch, 2, -1])
    # generate a 1D idx for the forward and backward direction
    scaler = torch.ones_like(sparse_idx).to(dev)
    scaler = scaler * n_nodes
    scaler_f = scaler
    scaler_f[:, 0] = 1
    scaler_r = torch.flip(scaler_f, [1])
    batch_offset = torch.arange(n_batch).unsqueeze(-1).expand([-1, n_nodes * k]) * n_nodes * k
    batch_offset = batch_offset.to(dev)
    sparse_idx_f = torch.sum(scaler_f * sparse_idx, 1) + batch_offset
    flat_idx_f = sparse_idx_f.view([-1])
    sparse_idx_r = torch.sum(scaler_r * sparse_idx, 1) + batch_offset
    flat_idx_r = sparse_idx_r.view([-1])
    # generate sparse tensors
    flat_h_E_f = h_E.view([n_batch * n_nodes * k, n_aa**2])
    reverse_h_E = h_E.transpose(-2, -1).contiguous()
    flat_h_E_r = reverse_h_E.view([n_batch * n_nodes * k, n_aa**2])
    sparse_etab_f = torch.sparse_coo_tensor(flat_idx_f.unsqueeze(0), flat_h_E_f,
                                            (n_batch * n_nodes * n_nodes, n_aa**2))
    count_f = torch.sparse_coo_tensor(flat_idx_f.unsqueeze(0), torch.ones_like(flat_idx_f),
                                      (n_batch * n_nodes * n_nodes, ))
    sparse_etab_r = torch.sparse_coo_tensor(flat_idx_r.unsqueeze(0), flat_h_E_r,
                                            (n_batch * n_nodes * n_nodes, n_aa**2))
    count_r = torch.sparse_coo_tensor(flat_idx_r.unsqueeze(0), torch.ones_like(flat_idx_r),
                                      (n_batch * n_nodes * n_nodes, ))
    # merge
    sparse_etab = sparse_etab_f + sparse_etab_r
    sparse_etab = sparse_etab.coalesce()
    count = count_f + count_r
    count = count.coalesce()

    # this step is very slow, but implementing something faster is probably a lot of work
    # requires pytorch 1.10 to be fast enough to be usable
    collect = sparse_etab.index_select(0, flat_idx_f).to_dense()
    weight = count.index_select(0, flat_idx_f).to_dense()

    flat_merged_etab = collect / weight.unsqueeze(-1)
    merged_etab = flat_merged_etab.view(h_E.shape)
    return merged_etab


def merge_duplicate_pairE_geometric(h_E, edge_index, orig_length=4):
    """ Sparse method to average pair energy tables across bidirectional edges with Torch Geometric.

    TERMinator edges are represented as two bidirectional edges, and to allow for
    communication between these edges we average the embeddings. In the case for
    pair energies, we transpose the tables to ensure that the pair energy table
    is symmetric upon inverse (e.g. the pair energy between i and j should be
    the same as the pair energy between j and i)

    This function assumes edge_index is sorted by columns, and will fail if
    this is not the case.

    Args
    ----
    h_E : torch.Tensor
        Pair energies in Torch Geometric sparse form
        Shape : n_edge x 400
    E_idx : torch.LongTensor
        Torch Geometric sparse edge indices
        Shape : 2 x n_edge

    Returns
    -------
    torch.Tensor
        Pair energies with merged energies for bidirectional edges
        Shape : n_edge x 400
    """
    num_nodes = edge_index.max() + 1
    row_idx = edge_index[0] + edge_index[1] * num_nodes
    col_idx = edge_index[1] + edge_index[0] * num_nodes
    internal_idx = torch.arange(edge_index.shape[1]).to(h_E.device)

    mapping = torch.zeros(max(row_idx.max(), col_idx.max()) + 1).long().to(h_E.device) - 1
    mapping[col_idx] = internal_idx

    reverse_idx = mapping[row_idx]
    mask = (reverse_idx >= 0)
    reverse_idx = reverse_idx[mask]

    reverse_h_E = h_E[mask]
    transpose_h_E = reverse_h_E.view([-1, 20, 20]).transpose(-1, -2).reshape([-1, h_E.shape[-1]])
    
    h_E[reverse_idx] = (h_E[reverse_idx] + transpose_h_E)/2

    return h_E


def expand_etab(etab, idxs):
    h = etab.shape[-1]
    tetab = etab.to(dtype=torch.float64)
    eidx = idxs.unsqueeze(-1).unsqueeze(-1).expand(etab.shape)
    netab = torch.zeros(tetab.shape[0], tetab.shape[1], tetab.shape[1], h, h, dtype=torch.float64, device=etab.device)
    netab.scatter_(2, eidx, tetab)
    cetab = netab.transpose(1,2).transpose(3,4)
    cetab.scatter_(2, eidx, tetab)
    return cetab

def batch_score_seq_batch(etab, seq_ints):
    h = etab.shape[-1]
    mask = torch.triu(torch.ones(etab.shape[0], etab.shape[1], etab.shape[1]), diagonal=0).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, h, h).to(etab.device)
    etab *= mask
    b, n, l = seq_ints.shape
    batch_etab = etab.unsqueeze(1).expand(b, n, l, l, h, h)
    j_batch_seq = seq_ints.unsqueeze(2).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, l, -1, h, 1)
    i_batch_seq = seq_ints.unsqueeze(3).unsqueeze(3).expand(b, n, l, l, 1)
    j_batch_eners = torch.gather(batch_etab, 5, j_batch_seq).squeeze(-1)
    batch_eners = torch.gather(j_batch_eners, 4, i_batch_seq).sum(dim=(2,3)).squeeze(-1)
    return batch_eners

def calc_eners_stability(etab, E_idx, sort_seqs, sort_nrgs, filter=True, expanded=False):
    if not expanded:
        etab = expand_etab(etab, E_idx)
    batch_scores = batch_score_seq_batch(etab, sort_seqs)
    ref_seqs = sort_seqs
    ref_energies = sort_nrgs
    if filter:
        mask = ref_energies != torch.nan
        ref_seqs = ref_seqs[mask]
        ref_energies = ref_energies[mask]
        batch_pred_E = batch_scores[mask]
        return batch_pred_E, ref_seqs, ref_energies
    else:
        return batch_scores, ref_seqs, ref_energies

def stability_loss_loop(base_etab, E_idx, data, max_tokens=20000, use_sc_mask=False, return_preds=False, return_norm=True):
    ''' Compute the correlation between etab's predicted energies and experimental stability energies.
    '''
    # TODO hacky fix to avoid problems when without SORTCERY data

    if data["sortcery_seqs"].numel() < 5:
        return 0, -1
    
    b, n, k, h = base_etab.shape
    h = int(np.sqrt(h))
    etab = base_etab.view(b, n, k, h, h)
    pad = (0, 2, 0, 2)
    etab = F.pad(etab, pad, "constant", 0)
    if n*data['sortcery_nrgs'].shape[1] > max_tokens:
        batch_size = int(max_tokens / n)
    else:
        batch_size = data['sortcery_nrgs'].shape[1]
    all_preds = []
    all_refs = []
    all_seqs = []
    for batch in range(0, data['sortcery_nrgs'].shape[1], batch_size):
        predicted_E, seqs, ref_energies = calc_eners_stability(etab, E_idx, data['sortcery_seqs'][:,batch:batch+batch_size], data['sortcery_nrgs'][:,batch:batch+batch_size])
        all_preds.append(predicted_E)
        all_refs.append(ref_energies)
        all_seqs.append(seqs)

    # normalize values around 0 for pearson correlation calculation
    predicted_E = torch.cat(all_preds, dim=0)
    ref_energies = torch.cat(all_refs, dim=0)
    all_seqs = torch.cat(all_seqs, dim=0)
    norm_pred = predicted_E - torch.mean(predicted_E) # n
    norm_ref = ref_energies - torch.mean(ref_energies) # n

    pearson = torch.sum(norm_pred * norm_ref) / (torch.sqrt(torch.sum(norm_pred**2)) * torch.sqrt(torch.sum(norm_ref**2)))
    if return_preds:
        if not return_norm:
            return -pearson, predicted_E, ref_energies, all_seqs
        return -pearson, norm_pred, norm_ref, all_seqs
    if torch.isnan(pearson):
        return 0, -1
    
    return -pearson, data['sortcery_nrgs'].shape[1] # scalar; negate, since we want to minimize our loss function


def stability_loss_loop_ddg(base_etab, E_idx, data, max_tokens=20000, use_sc_mask=False, return_preds=False, return_norm=True):
    ''' Compute the correlation between etab's predicted energies and experimental stability energies.
    '''
    # TODO hacky fix to avoid problems when without SORTCERY data

    if data["sortcery_seqs"].numel() < 5:
        return 0, -1
    
    b, n, k, h = base_etab.shape
    h = int(np.sqrt(h))
    etab = base_etab.view(b, n, k, h, h)
    pad = (0, 2, 0, 2)
    etab = F.pad(etab, pad, "constant", 0)

    ## Add WT
    seqs = torch.cat([data['seqs'].unsqueeze(1), data['sortcery_seqs']], dim=1)
    nrgs = F.pad(data['sortcery_nrgs'], (1, 0), "constant", 0)

    if n*nrgs.shape[1] > max_tokens:
        batch_size = int(max_tokens / n)
    else:
        batch_size = nrgs.shape[1]
    all_preds = []
    all_refs = []
    all_seqs = []
    
    for batch in range(0, nrgs.shape[1], batch_size):
        predicted_E, cur_seqs, ref_energies = calc_eners_stability(etab, E_idx, seqs[:,batch:batch+batch_size], nrgs[:,batch:batch+batch_size])
        all_preds.append(predicted_E)
        all_refs.append(ref_energies)
        all_seqs.append(cur_seqs)

    predicted_E = torch.cat(all_preds, dim=0)
    ref_energies = torch.cat(all_refs, dim=0) 
    all_seqs = torch.cat(all_seqs, dim=0)

    # Normalize to WT
    predicted_E = predicted_E[1:] - predicted_E[0]
    ref_energies = ref_energies[1:]
    all_seqs = all_seqs[1:]

    # Normalize values around 0 for pearson correlation calculation
    norm_pred = predicted_E - torch.mean(predicted_E) # n
    norm_ref = ref_energies - torch.mean(ref_energies) # n

    pearson = torch.sum(norm_pred * norm_ref) / (torch.sqrt(torch.sum(norm_pred**2)) * torch.sqrt(torch.sum(norm_ref**2)))
    if return_preds:
        if not return_norm:
            return -pearson, predicted_E, ref_energies, all_seqs
        return -pearson, norm_pred, norm_ref, all_seqs
    if torch.isnan(pearson):
        return 0, -1
    
    return -pearson, data['sortcery_nrgs'].shape[1] # scalar; negate, since we want to minimize our loss function

def setup_etab(etab, E_idx):
    b, n, k, h = etab.shape
    h = int(np.sqrt(h))
    etab = etab.view(b, n, k, h, h)
    pad = (0, 2, 0, 2)
    etab = F.pad(etab, pad, "constant", 0)
    etab = expand_etab(etab, E_idx)
    return etab

def stability_loss_diff_loop(base_etab, E_idx, data, etab_nolig, E_idx_nolig, max_tokens=20000, use_sc_mask=False, return_preds=False, return_norm=True, ddg=False):
    ''' Compute the correlation between etab's predicted energies and experimental stability energies.
    '''
    # TODO hacky fix to avoid problems when without SORTCERY data

    if data["sortcery_seqs"].numel() < 5:
        return 0, -1
    
    n = base_etab.shape[1]
    etab = setup_etab(base_etab, E_idx)
    nolig_etab = setup_etab(etab_nolig, E_idx_nolig)
    etab = etab - nolig_etab

    seqs = torch.cat([data['seqs'].unsqueeze(1), data['sortcery_seqs']], dim=1)
    nrgs = F.pad(data['sortcery_nrgs'], (1, 0), "constant", 0)
    
    if n*data['sortcery_nrgs'].shape[1] > max_tokens:
        batch_size = int(max_tokens / n)
    else:
        batch_size = data['sortcery_nrgs'].shape[1]
    all_preds = []
    all_refs = []
    all_seqs = []
    for batch in range(0, data['sortcery_nrgs'].shape[1], batch_size):
        predicted_E, cur_seqs, ref_energies = calc_eners_stability(etab, E_idx, seqs[:,batch:batch+batch_size], nrgs[:,batch:batch+batch_size], expanded=True)
        all_preds.append(predicted_E)
        all_refs.append(ref_energies)
        all_seqs.append(cur_seqs)

    predicted_E = torch.cat(all_preds, dim=0)
    ref_energies = torch.cat(all_refs, dim=0) 
    all_seqs = torch.cat(all_seqs, dim=0)

    # Normalize to WT
    if ddg:
        predicted_E = predicted_E[1:] - predicted_E[0]
    else:
        predicted_E = predicted_E[1:]
    ref_energies = ref_energies[1:]
    all_seqs = all_seqs[1:]

    # Normalize values around 0 for pearson correlation calculation
    norm_pred = predicted_E - torch.mean(predicted_E) # n
    norm_ref = ref_energies - torch.mean(ref_energies) # n

    pearson = torch.sum(norm_pred * norm_ref) / (torch.sqrt(torch.sum(norm_pred**2)) * torch.sqrt(torch.sum(norm_ref**2)))
    if return_preds:
        if not return_norm:
            return -pearson, predicted_E, ref_energies, all_seqs
        return -pearson, norm_pred, norm_ref, all_seqs
    if torch.isnan(pearson):
        return 0, -1
    return -pearson, data['sortcery_nrgs'].shape[1] # scalar; negate, since we want to minimize our loss function

# zero is used as padding
AA_to_int = {
    'A': 1,
    'ALA': 1,
    'C': 2,
    'CYS': 2,
    'D': 3,
    'ASP': 3,
    'E': 4,
    'GLU': 4,
    'F': 5,
    'PHE': 5,
    'G': 6,
    'GLY': 6,
    'H': 7,
    'HIS': 7,
    'I': 8,
    'ILE': 8,
    'K': 9,
    'LYS': 9,
    'L': 10,
    'LEU': 10,
    'M': 11,
    'MET': 11,
    'N': 12,
    'ASN': 12,
    'P': 13,
    'PRO': 13,
    'Q': 14,
    'GLN': 14,
    'R': 15,
    'ARG': 15,
    'S': 16,
    'SER': 16,
    'T': 17,
    'THR': 17,
    'V': 18,
    'VAL': 18,
    'W': 19,
    'TRP': 19,
    'Y': 20,
    'TYR': 20,
    '-': 21,
    'X': 22
}

esm_list = [ 5, 23, 13,  9, 18,  6, 21, 12, 15,  4, 20, 17, 14, 16, 10,  8, 11,
          7, 22, 19, 30, 24] # Alphabetical
esm_encodings = {}
for i, e in enumerate(esm_list):
    esm_encodings[i] = e
esm_decodings = {}
for i, e in enumerate(esm_list):
    esm_decodings[e] = i

AA_to_int = {key: val - 1 for key, val in AA_to_int.items()}

int_to_AA = {y: x for x, y in AA_to_int.items() if len(x) == 1}

int_to_3lt_AA = {y: x for x, y in AA_to_int.items() if len(x) == 3}

def seq_to_ints(sequence):
    """
    Given a string of one-letter encoded AAs, return its corresponding integer encoding
    """
    return [AA_to_int[residue] for residue in sequence]


def ints_to_seq(int_list):
    return [int_to_AA[i] if i in int_to_AA.keys() else 'X' for i in int_list]

def aa_three_to_one(residue):
    return int_to_AA[AA_to_int[residue]]

def esm_convert(seq):
    return [esm_encodings[s] for s in seq]

def esm_deconvert(seq):
    return [esm_decodings[s] for s in seq]

def ints_to_seq_torch(seq):
    return "".join(ints_to_seq(seq.cpu().numpy()))

def esm_ints_to_seq_torch(seq):
    return "".join(ints_to_seq(esm_deconvert(seq)))

def ints_to_seq_normal(seq):
    return "".join(ints_to_seq(seq))