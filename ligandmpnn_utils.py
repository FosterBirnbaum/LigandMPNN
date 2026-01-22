import pandas as pd
import pickle
import os
import torch
import numpy as np
import copy
from collections import defaultdict
import json
import omegaconf
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
from data_utils import parse_PDB, parse_PDB_seq_only, get_nearest_neighbours, alphabet
from model_utils import nlcpl
import etab_utils

checkpoints = {
    "protein_mpnn": "./model_params/proteinmpnn_v_48_020.pt",
    "ligand_mpnn": "./model_params/ligandmpnn_v_32_010_25.pt",
    "per_residue_label_membrane_mpnn": "./model_params/per_residue_label_membrane_mpnn_v_48_020.pt",
    "global_label_membrane_mpnn": "./model_params/global_label_membrane_mpnn_v_48_020.pt",
    "soluble_mpnn": "./model_params/solublempnn_v_48_020.pt",  
    'ligand_mpnn_retrained': "/home/fosterb/LigandMPNN/model_params/ligandmpnn_default_nomixed_cont/model_weights/epoch_best.pt",
    'ligand_mpnn_potts': "/home/fosterb/LigandMPNN/model_params/ligandmpnn_potts_nomixed_cont/model_weights/epoch_best.pt",
    'ligand_mpnn_potts_v2': "/mnt/shared3/fosterb/pmpnn_runs/ligandmpnn_potts_4enc/model_weights/epoch_best.pt"
}

restype_3to1 = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
    }
from Bio.PDB import PDBParser, PPBuilder, Polypeptide
import pandas as pd

def aa_or_x(residue):
    """Return 1-letter AA code or 'X' for non-canonical residues."""
    try:
        aa = Polypeptide.three_to_one(residue.get_resname())
    except KeyError:
        aa = "X"
    return aa

def generate_mut_dataframe(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_path)
    model = structure[0]

    # Identify protein chains (chains containing at least one residue)
    protein_chains = []
    for chain in model:
        for res in chain:
            if res.id[0] == " ":  # exclude heteroatoms, water
                protein_chains.append(chain.id)
                break
    protein_chains = sorted(set(protein_chains))

    # Build per-chain sequence using AA=X for noncanonical residues
    chain_sequences = {}
    for chain in model:
        if chain.id not in protein_chains:
            continue
        seq = []
        for res in chain:
            if res.id[0] != " ":
                pass
                # continue
            seq.append(aa_or_x(res))
        chain_sequences[chain.id] = "".join(seq)

    # Standard AA list
    aas = list("ACDEFGHIKLMNPQRSTVWY")
    WT_name = os.path.basename(pdb_path).split('.')[0]
    rows = []
    for chain_id, sequence in chain_sequences.items():
        for pos, wt_res in enumerate(sequence, start=0):

            for mut in aas:
                if mut == wt_res:
                    continue

                mut_type = f"{wt_res}{pos}{mut}"
                sig = f"{WT_name}_{chain_id}_{mut_type}"

                rows.append({
                    "WT_name": WT_name,
                    "mut_type": mut_type,
                    "ddG_ML": 0,
                    "chain": chain_id,
                    "chain_list": protein_chains,
                    "sig": sig
                })

    return pd.DataFrame(rows)

def loss_nll(S, log_probs, mask):
    """ Negative log probabilities """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)), S.contiguous().view(-1)
    ).view(S.size())
    S_argmaxed = torch.argmax(log_probs,-1) #[B, L]
    true_false = (S == S_argmaxed).float()
    loss_masked = loss * mask
    return loss, loss_masked, true_false


def featurize(
    input_dict,
    cutoff_for_score=8.0,
    use_atom_context=True,
    number_of_ligand_atoms=16,
    model_type="protein_mpnn"):
    output_dict = {}
    if model_type == "ligand_mpnn":
        mask = input_dict["mask"]
        Y = input_dict["Y"]
        Y_t = input_dict["Y_t"]
        Y_m = input_dict["Y_m"]
        N = input_dict["X"][:, 0, :]
        CA = input_dict["X"][:, 1, :]
        C = input_dict["X"][:, 2, :]
        b = CA - N
        c = C - CA
        a = torch.cross(b, c, axis=-1)
        CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
        Y, Y_t, Y_m, D_XY = get_nearest_neighbours(
            CB, mask, Y, Y_t, Y_m, number_of_ligand_atoms
        )
        mask_XY = (D_XY < cutoff_for_score) * mask * Y_m[:, 0]
        output_dict["mask_XY"] = mask_XY[None,]
        if "side_chain_mask" in list(input_dict):
            output_dict["side_chain_mask"] = input_dict["side_chain_mask"][None,]
        output_dict["Y"] = Y[None,]
        output_dict["Y_t"] = Y_t[None,]
        output_dict["Y_m"] = Y_m[None,]
        if not use_atom_context:
            output_dict["Y_m"] = 0.0 * output_dict["Y_m"]
    elif (
        model_type == "per_residue_label_membrane_mpnn"
        or model_type == "global_label_membrane_mpnn"
    ):
        output_dict["membrane_per_residue_labels"] = input_dict[
            "membrane_per_residue_labels"
        ][None,]

    R_idx_list = []
    count = 0
    R_idx_prev = -100000
    for R_idx in list(input_dict["R_idx"]):
        if R_idx_prev == R_idx:
            count += 1
        R_idx_list.append(R_idx + count)
        R_idx_prev = R_idx
    R_idx_renumbered = torch.tensor(R_idx_list, device=R_idx.device)
    output_dict["R_idx"] = R_idx_renumbered[None,]
    output_dict["R_idx_original"] = input_dict["R_idx"][None,]
    output_dict["chain_labels"] = input_dict["chain_labels"][None,]
    output_dict["S"] = input_dict["S"][None,]
    output_dict["chain_mask"] = input_dict["chain_mask"][None,]
    output_dict["mask"] = input_dict["mask"][None,]

    output_dict["X"] = input_dict["X"][None,]

    if "xyz_37" in list(input_dict):
        output_dict["xyz_37"] = input_dict["xyz_37"][None,]
        output_dict["xyz_37_m"] = input_dict["xyz_37_m"][None,]

    return output_dict

def get_log_probs(input_pdb, chain_list, model, device, ligand_mpnn_use_side_chain_context,
                  parse_atoms_with_zero_occupancy, transmembrane_buried, transmembrane_interface,
                  model_type, ligand_mpnn_cutoff_for_score, ligand_mpnn_use_atom_context, atom_context_num,
                  temperature, updated_alist, nolig=False):
    
    # make protein dict
    protein_dict, backbone, other_atoms, icodes, _ = parse_PDB(
                input_pdb,
                device=device,
                chains=chain_list,
                parse_all_atoms=ligand_mpnn_use_side_chain_context,
                parse_atoms_with_zero_occupancy=parse_atoms_with_zero_occupancy,
                updated_alist=updated_alist,
                modify_list=['3MI3']
            )
    if nolig:
        protein_dict["Y"] = torch.zeros((1, 3), dtype=torch.float32).to(protein_dict["Y"].device)
        protein_dict["Y_t"] = torch.zeros((1), dtype=torch.float32).to(protein_dict["Y_t"].device)
        protein_dict["Y_m"] = torch.zeros((1), dtype=torch.float32).to(protein_dict["Y_m"].device)

    # make chain_letter + residue_idx + insertion_code mapping to integers
    R_idx_list = list(protein_dict["R_idx"].cpu().numpy())  # residue indices
    chain_letters_list = list(protein_dict["chain_letters"])  # chain letters
    encoded_residues = []
    for i, R_idx_item in enumerate(R_idx_list):
        tmp = str(chain_letters_list[i]) + str(R_idx_item) + icodes[i]
        encoded_residues.append(tmp)
    encoded_residue_dict = dict(zip(encoded_residues, range(len(encoded_residues))))
    encoded_residue_dict_rev = dict(
        zip(list(range(len(encoded_residues))), encoded_residues)
    )

    bias_AA_per_residue = torch.zeros(
        [len(encoded_residues), 21], device=device, dtype=torch.float32
    )

    fixed_positions = torch.tensor(
        [int(True) for item in encoded_residues],
        device=device,
    )
    redesigned_positions = torch.tensor(
        [int(False) for item in encoded_residues],
        device=device,
    )

    # specify which residues are buried for checkpoint_per_residue_label_membrane_mpnn model
    if transmembrane_buried:
        buried_residues = [item for item in transmembrane_buried.split()]
        buried_positions = torch.tensor(
            [int(item in buried_residues) for item in encoded_residues],
            device=device,
        )
    else:
        buried_positions = torch.zeros_like(fixed_positions)

    if transmembrane_interface:
        interface_residues = [item for item in transmembrane_interface.split()]
        interface_positions = torch.tensor(
            [int(item in interface_residues) for item in encoded_residues],
            device=device,
        )
    else:
        interface_positions = torch.zeros_like(fixed_positions)
    protein_dict["membrane_per_residue_labels"] = 2 * buried_positions * (
        1 - interface_positions
    ) + 1 * interface_positions * (1 - buried_positions)
    
    # create chain_mask
    chains_to_design_list = protein_dict["chain_letters"]
    chain_mask = torch.tensor(
        np.array(
            [
                item in chains_to_design_list
                for item in protein_dict["chain_letters"]
            ],
            dtype=np.int32,
        ),
        device=device,
    )
    protein_dict["chain_mask"] = chain_mask

    # featurize
    feature_dict = featurize(
        protein_dict,
        cutoff_for_score=ligand_mpnn_cutoff_for_score,
        use_atom_context=ligand_mpnn_use_atom_context,
        number_of_ligand_atoms=atom_context_num,
        model_type=model_type,
    )
    feature_dict["batch_size"] = 1
    feature_dict["temperature"] = temperature
    B, L, _, _ = feature_dict["X"].shape  # batch size should be 1 for now.
    omit_AA = torch.tensor(
        np.array([False for AA in alphabet]).astype(np.float32),
        device=device,
    )
    omit_AA_per_residue = torch.zeros(
        [len(encoded_residues), 21], device=device, dtype=torch.float32
    )
    bias_AA = torch.zeros([21], device=device, dtype=torch.float32)
    bias_AA_per_residue = torch.zeros(
        [len(encoded_residues), 21], device=device, dtype=torch.float32
    )
    feature_dict["bias"] = (
                    (-1e8 * omit_AA[None, None, :] + bias_AA).repeat([1, L, 1])
                    + bias_AA_per_residue[None]
                    - 1e8 * omit_AA_per_residue[None]
                )
    feature_dict["symmetry_residues"] = [[]]
    feature_dict["symmetry_weights"] = [[]]
    feature_dict["randn"] = torch.randn(
        [feature_dict["batch_size"], feature_dict["mask"].shape[1]],
        device=device,
    )

    with torch.no_grad():
        output_dict = model.sample(feature_dict)
        log_probs, etab, E_idx = model(feature_dict)

    _, nlll_loss, nsr = loss_nll(feature_dict["S"].to(dtype=torch.int64), output_dict["log_probs"], feature_dict["mask"])
    
    if etab is not None:
        nlcpl_loss, _ = nlcpl(etab, E_idx, feature_dict["S"].to(dtype=torch.int64), feature_dict["mask"])
        nlcpl_loss = nlcpl_loss.cpu().item()
    else:
        nlcpl_loss = np.nan
        
    return log_probs, nlll_loss, nsr, etab, E_idx, nlcpl_loss

def is_float(s):
    """
    Checks if the input string 's' can be converted to a float.
    Returns the converted float if it can, None otherwise.
    """
    try:
        return float(s)
    except ValueError:
        return None

def process_data(cfg, pdb_list):
    """
    Process data settings for energy prediction.

    Parameters
    ----------
    cfg : OmegaConf object
    pdb_list : list of pdb names

    Returns
    -------
    dataset_settings : dict of dicts
        Processed dataset settings per pdb
    chain_lens_dicts : dict of lists
        Chain lengths per pdb
    pdb_list : list of pdb names
    binding_energy_chains : None or dict of chain list pairs for binding energy calculation

    """

    # If predicting binding energies, load information about chain separation
    if cfg.inference.binding_energy_json:
        if type(cfg.inference.binding_energy_json) in [dict, omegaconf.dictconfig.DictConfig]:
            binding_energy_chains = cfg.inference.binding_energy_json
        else:
            with open(cfg.inference.binding_energy_json, 'r') as f:
                binding_energy_chains = json.load(f)
            for pdb in pdb_list:
                if not pdb in binding_energy_chains:
                    binding_energy_chains[pdb] = None
    else:
        binding_energy_chains = None

    # Set up data structures
    mutant_data = {'pdb': [], 'sequences': [], 'partitioned_sequences': [], 'ddG_expt': [], 'mut_chains': []}
    chain_lens_dicts = {}
    mut_alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    wt_seqs = {}
    partitioned_wt_seqs = {}
    # Load mutant sequence information
    if cfg.mutant_fasta is not None: # Predict energies for provided mutant sequences from a FASTA file
        with open(cfg.mutant_fasta, 'r') as f:
            mutant_seq_lines = f.readlines()
        mutant_seqs = defaultdict(list)
        for pdb, line in zip(mutant_seq_lines[::2], mutant_seq_lines[1::2]):
            mutant_seqs[pdb.strip().split('|')[0].strip('>')].append((pdb.strip(), line.strip()))

        for pdb in pdb_list:
            # Gather information about wild-type sequence
            wt_info = parse_PDB_seq_only(os.path.join(cfg.input_dir, pdb + '.pdb'), skip_gaps=cfg.inference.skip_gaps)
            for header, seq in mutant_seqs[pdb]:
                header = header.strip('>')
                # Parse mutant sequence header
                header_parts = header.split('|')
                assert len(header_parts) <= 3, "Header information cannot exceed 3 '|' parts"
                mut_chains = None
                ddG_expt = None
                if len(header_parts) == 2:
                    ddG_expt = is_float(header_parts[1])
                    if not ddG_expt:
                        mut_chains = header_parts[1]
                elif len(header_parts) == 3:
                    mut_chains = header_parts[1]
                    ddG_expt = is_float(header_parts[2])
                if not ddG_expt: ddG_expt = np.nan

                # Create full mutant sequence
                mut_seq = []
                wt_seq = []
                if mut_chains: # If chain info in header, processes provided sequence accordingly
                    mut_chains = mut_chains.split(':')
                else: # Assume mutant sequence provided has all chains present
                    assert len(wt_info['chain_order']) == len(seq.split(':')), "If chains not specified, mutant sequence must contain information on all chains"
                    mut_chains = wt_info['chain_order']
                mut_seq_dict = {chain: chain_seq for chain, chain_seq in zip(mut_chains, seq.split(':'))}

                for chain in wt_info['chain_order']:
                    if chain in mut_seq_dict: # Use mutant sequence
                        assert len(mut_seq_dict[chain]) == len(wt_info[f'seq_chain_{chain}']), "Mutant sequence length must match wildtype sequence length"
                        # Check mutant seq to ensure mutations are all canonical amino acids
                        for wc, mc in zip(wt_info[f'seq_chain_{chain}'], mut_seq_dict[chain]):
                            if wc != mc: assert mc in mut_alphabet, "Mutation must be one of 20 canonical amino acids"
                        mut_seq.append((chain, mut_seq_dict[chain]))
                    else: # Use wildtype sequence
                        mut_seq.append((chain, wt_info[f'seq_chain_{chain}']))
                    wt_seq.append((chain, wt_info[f'seq_chain_{chain}']))
                mutant_data['pdb'].append(pdb)
                mutant_data['sequences'].append(mut_seq)
                mutant_data['ddG_expt'].append(ddG_expt)
                mutant_data['mut_chains'].append(':'.join(mut_chains))
            chain_lens_dicts[pdb] = {chain: len(chain_seq) for chain, chain_seq in mutant_data['sequences'][-1]}
            wt_seqs[pdb] = wt_seq

    elif cfg.mutant_csv is not None: # Predict energies for provided mutant sequences from a CSV file
        mutant_df = pd.read_csv(cfg.mutant_csv)
        assert all(col in mutant_df.columns for col in ['pdb', 'chain', 'mut_type']), "CSV must contain 'pdb', 'chain', and 'mut_type' columns"
        if not 'ddG_expt' in mutant_df.columns:
            mutant_df['ddG_expt'] = [np.nan] * len(mutant_df)
        for pdb in mutant_df['pdb'].unique():
            pdb_df = mutant_df[mutant_df['pdb'] == pdb]
            wt_info = parse_PDB_seq_only(os.path.join(cfg.input_dir, pdb + '.pdb'), skip_gaps=cfg.inference.skip_gaps)
            for chain_list, mut_type_list, ddG_expt in zip(pdb_df['chain'], pdb_df['mut_type'], pdb_df['ddG_expt']):
                mut_type_dict = defaultdict(list)
                for chain, mut_type in zip(chain_list.split(':'), mut_type_list.split(':')):
                    mut_type_dict[chain].append(mut_type)
                mut_seq = []
                wt_seq = []
                for chain in wt_info['chain_order']:
                    mut_chain = copy.deepcopy(wt_info[f'seq_chain_{chain}'])
                    if len(mut_type_dict[chain]) > 0: # Use mutant sequence
                        for mut_type in mut_type_dict[chain]:
                            wt, pos, mut = mut_type[0], int(mut_type[1:-1]), mut_type[-1]
                            assert wt == mut_chain[pos], "Mutation information must match wildtype sequence at the mutation position"
                            assert mut in mut_alphabet, "Mutation must be one of 20 canonical amino acids"
                            mut_chain = mut_chain[:pos] + mut + mut_chain[pos+1:]
                        mut_seq.append((chain, mut_chain))
                    else: # Use wildtype sequence
                        mut_seq.append((chain, wt_info[f'seq_chain_{chain}']))
                    wt_seq.append((chain, wt_info[f'seq_chain_{chain}']))
                mutant_data['pdb'].append(pdb)
                mutant_data['sequences'].append(mut_seq)
                mutant_data['ddG_expt'].append(ddG_expt)
                mutant_data['mut_chains'].append(chain_list)
            chain_lens_dicts[pdb] = {chain: len(chain_seq) for chain, chain_seq in mutant_data['sequences'][-1]}
            wt_seqs[pdb] = wt_seq

    else: # Do a DMS screen of all single mutants
        for pdb in pdb_list:
            wt_info = parse_PDB_seq_only(os.path.join(cfg.input_dir, pdb + '.pdb'), skip_gaps=cfg.inference.skip_gaps)
            wt_chains = [(chain, wt_info[f'seq_chain_{chain}']) for chain in wt_info['chain_order']]
            for i_chain, chain in enumerate(wt_info['chain_order']):
                mut_seq = ""
                for i, wtAA in enumerate(wt_info[f'seq_chain_{chain}']):
                    if wtAA != '-':
                        for mutAA in mut_alphabet:
                            if mutAA != wtAA:
                                mut_seq = copy.deepcopy(wt_info[f'seq_chain_{chain}'])
                                mut_seq = mut_seq[:i] + mutAA + mut_seq[i+1:]
                                mutant_data['pdb'].append(pdb)
                                full_mut_seq = copy.deepcopy(wt_chains)
                                full_mut_seq[i_chain] = (chain, mut_seq)
                                mutant_data['sequences'].append(full_mut_seq)
                                mutant_data['ddG_expt'].append(np.nan)
                                mutant_data['mut_chains'].append(chain)
            chain_lens_dicts[pdb] = {chain: len(chain_seq) for chain, chain_seq in mutant_data['sequences'][-1]}
            wt_seqs[pdb] = wt_chains

    if binding_energy_chains: # Split sequences into separate chains if requested for binding prediction
        for pdb, seq in zip(mutant_data['pdb'], mutant_data['sequences']):
            assert pdb in binding_energy_chains.keys(), "To calculate binding energies, chain partition information must be present for each structure"
            all_chains = []
            for partition in binding_energy_chains[pdb]:
                all_chains += partition
            assert sorted(all_chains) == sorted([chain for chain, _ in mutant_data['sequences'][0]]), "Chain partitions must include all chains in structure"
            partitioned_sequences = []
            for partition in binding_energy_chains[pdb]:
                partitioned_sequences.append("".join([chain_seq for chain, chain_seq in seq if chain in partition]))
            mutant_data['partitioned_sequences'].append(partitioned_sequences)

        for pdb, wt_seq in wt_seqs.items():
            partitioned_wt_seqs[pdb] = []
            for partition in binding_energy_chains[pdb]:
                partitioned_wt_seqs[pdb].append("".join([chain_seq for chain, chain_seq in wt_seq if chain in partition]))
            wt_seqs[pdb] = "".join([chain_seq for _, chain_seq in wt_seq])

    else:
        mutant_data['partitioned_sequences'] = [None] * len(mutant_data['sequences'])
        for pdb, wt_seq in wt_seqs.items():
            wt_seqs[pdb] = "".join([chain_seq for _, chain_seq in wt_seq])
            partitioned_wt_seqs[pdb] = None
    # Save mutant sequences and energies to tensors
    for i_mut in range(len(mutant_data['sequences'])):
        mutant_data['sequences'][i_mut] = "".join([chain_seq for _, chain_seq in mutant_data['sequences'][i_mut]])
    
    return pd.DataFrame(mutant_data), chain_lens_dicts, wt_seqs, partitioned_wt_seqs, pdb_list, binding_energy_chains

def featurize(
    input_dict,
    cutoff_for_score=8.0,
    use_atom_context=True,
    number_of_ligand_atoms=16,
    model_type="protein_mpnn"):
    output_dict = {}
    if model_type == "ligand_mpnn":
        mask = input_dict["mask"]
        Y = input_dict["Y"]
        Y_t = input_dict["Y_t"]
        Y_m = input_dict["Y_m"]
        N = input_dict["X"][:, 0, :]
        CA = input_dict["X"][:, 1, :]
        C = input_dict["X"][:, 2, :]
        b = CA - N
        c = C - CA
        a = torch.cross(b, c, axis=-1)
        CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
        Y, Y_t, Y_m, D_XY = get_nearest_neighbours(
            CB, mask, Y, Y_t, Y_m, number_of_ligand_atoms
        )
        mask_XY = (D_XY < cutoff_for_score) * mask * Y_m[:, 0]
        output_dict["mask_XY"] = mask_XY[None,]
        if "side_chain_mask" in list(input_dict):
            output_dict["side_chain_mask"] = input_dict["side_chain_mask"][None,]
        output_dict["Y"] = Y[None,]
        output_dict["Y_t"] = Y_t[None,]
        output_dict["Y_m"] = Y_m[None,]
        if not use_atom_context:
            output_dict["Y_m"] = 0.0 * output_dict["Y_m"]
    elif (
        model_type == "per_residue_label_membrane_mpnn"
        or model_type == "global_label_membrane_mpnn"
    ):
        output_dict["membrane_per_residue_labels"] = input_dict[
            "membrane_per_residue_labels"
        ][None,]

    R_idx_list = []
    count = 0
    R_idx_prev = -100000
    for R_idx in list(input_dict["R_idx"]):
        if R_idx_prev == R_idx:
            count += 1
        R_idx_list.append(R_idx + count)
        R_idx_prev = R_idx
    R_idx_renumbered = torch.tensor(R_idx_list, device=R_idx.device)
    output_dict["R_idx"] = R_idx_renumbered[None,]
    output_dict["R_idx_original"] = input_dict["R_idx"][None,]
    output_dict["chain_labels"] = input_dict["chain_labels"][None,]
    output_dict["S"] = input_dict["S"][None,]
    output_dict["chain_mask"] = input_dict["chain_mask"][None,]
    output_dict["mask"] = input_dict["mask"][None,]

    output_dict["X"] = input_dict["X"][None,]

    if "xyz_37" in list(input_dict):
        output_dict["xyz_37"] = input_dict["xyz_37"][None,]
        output_dict["xyz_37_m"] = input_dict["xyz_37_m"][None,]

    return output_dict

def get_log_probs_colab(input_pdb, chain_list, model, cfg):
  
    device = cfg.dev
    
    # make protein dict
    protein_dict, backbone, other_atoms, icodes, _ = parse_PDB(
                input_pdb,
                device=device,
                chains=chain_list,
                parse_all_atoms=cfg.inference.ligand_mpnn_use_side_chain_context,
                parse_atoms_with_zero_occupancy=cfg.inference.parse_atoms_with_zero_occupancy,
                updated_alist=cfg.inference.updated_alist,
            )
    if cfg.inference.nolig:
        protein_dict["Y"] = torch.zeros((1, 3), dtype=torch.float32).to(protein_dict["Y"].device)
        protein_dict["Y_t"] = torch.zeros((1), dtype=torch.float32).to(protein_dict["Y_t"].device)
        protein_dict["Y_m"] = torch.zeros((1), dtype=torch.float32).to(protein_dict["Y_m"].device)

    # make chain_letter + residue_idx + insertion_code mapping to integers
    R_idx_list = list(protein_dict["R_idx"].cpu().numpy())  # residue indices
    chain_letters_list = list(protein_dict["chain_letters"])  # chain letters
    encoded_residues = []
    for i, R_idx_item in enumerate(R_idx_list):
        tmp = str(chain_letters_list[i]) + str(R_idx_item) + icodes[i]
        encoded_residues.append(tmp)
    encoded_residue_dict = dict(zip(encoded_residues, range(len(encoded_residues))))
    encoded_residue_dict_rev = dict(
        zip(list(range(len(encoded_residues))), encoded_residues)
    )

    bias_AA_per_residue = torch.zeros(
        [len(encoded_residues), cfg.model.vocab], device=device, dtype=torch.float32
    )

    fixed_positions = torch.tensor(
        [int(True) for item in encoded_residues],
        device=device,
    )
    redesigned_positions = torch.tensor(
        [int(False) for item in encoded_residues],
        device=device,
    )

    # specify which residues are buried for checkpoint_per_residue_label_membrane_mpnn model
    if cfg.inference.transmembrane_buried:
        buried_residues = [item for item in cfg.inference.transmembrane_buried.split()]
        buried_positions = torch.tensor(
            [int(item in buried_residues) for item in encoded_residues],
            device=device,
        )
    else:
        buried_positions = torch.zeros_like(fixed_positions)

    if cfg.inference.transmembrane_interface:
        interface_residues = [item for item in cfg.inference.transmembrane_interface.split()]
        interface_positions = torch.tensor(
            [int(item in interface_residues) for item in encoded_residues],
            device=device,
        )
    else:
        interface_positions = torch.zeros_like(fixed_positions)
    protein_dict["membrane_per_residue_labels"] = 2 * buried_positions * (
        1 - interface_positions
    ) + 1 * interface_positions * (1 - buried_positions)

    # create chain_mask
    chains_to_design_list = protein_dict["chain_letters"]
    chain_mask = torch.tensor(
        np.array(
            [
                item in chains_to_design_list
                for item in protein_dict["chain_letters"]
            ],
            dtype=np.int32,
        ),
        device=device,
    )
    protein_dict["chain_mask"] = chain_mask

    # featurize
    feature_dict = featurize(
        protein_dict,
        cutoff_for_score=cfg.inference.ligand_mpnn_cutoff_for_score,
        use_atom_context=cfg.inference.ligand_mpnn_use_atom_context,
        number_of_ligand_atoms=cfg.inference.atom_context_num,
        model_type=cfg.model.model_type,
    )
    feature_dict["batch_size"] = 1
    B, L, _, _ = feature_dict["X"].shape  # batch size should be 1 for now.
    omit_AA = torch.tensor(
        np.array([False for AA in alphabet]).astype(np.float32),
        device=device,
    )
    omit_AA_per_residue = torch.zeros(
        [len(encoded_residues), cfg.model.vocab], device=device, dtype=torch.float32
    )
    bias_AA = torch.zeros([cfg.model.vocab], device=device, dtype=torch.float32)
    bias_AA_per_residue = torch.zeros(
        [len(encoded_residues), cfg.model.vocab], device=device, dtype=torch.float32
    )
    feature_dict["bias"] = (
                    (-1e8 * omit_AA[None, None, :] + bias_AA).repeat([1, L, 1])
                    + bias_AA_per_residue[None]
                    - 1e8 * omit_AA_per_residue[None]
                )
    feature_dict["symmetry_residues"] = [[]]
    feature_dict["symmetry_weights"] = [[]]
    feature_dict["randn"] = torch.randn(
        [feature_dict["batch_size"], feature_dict["mask"].shape[1]],
        device=device,
    )

    with torch.no_grad():
        log_probs, etab, E_idx = model(feature_dict)
  
    return log_probs, etab, E_idx


def score_seqs_potts(etab, E_idx, wt_seq, cfg, nrgs, seqs, partition=None):
    """
    Score sequences using the energy table.

    Parameters
    ----------  
    etab : torch.Tensor, shape (1, L, 400)
        Energy table from PottsMPNN model
    E_idx : torch.Tensor, shape (1, L, k)
        Edge index table from PottsMPNN model
    wt_seq : str, shape (L,)
        Wildtype sequence
    model : PottsMPNN model
        Model with which to score sequences
    cfg : omegaconf
        Config object
    pdb_data : dict
        dict with PDB information
    nrgs : list of shape (N,)
        Mutant energy information
    seqs : list of shape (N, L)
        Mutant sequence information
    partition : list (optional, default None)
        list of chains to analyze
    
    Returns
    -------
    scores : torch.Tensor, shape (N,)
        Scores for each sequence.
    scored_seqs : torch.Tensor, shape (N, L)
        Scored sequences
    reference_scores : torch.Tensor, shape (N,)
        References for scored sequences
    """
    
    # Run energy prediction according to config
    # Use wildtype as reference energy
    nrgs = np.insert(nrgs, 0, 0.0)
    seqs = np.insert(seqs, 0, wt_seq)
    # Transform nrgs and seqs to tensors
    nrgs = torch.from_numpy(np.array(nrgs)).to(dtype=torch.float32, device=cfg.dev).unsqueeze(0)
    seqs = torch.stack([etab_utils.seq_to_tensor(seq) for seq in seqs]).to(dtype=torch.int64, device=cfg.dev).unsqueeze(0)

    if etab.size(1)*nrgs.shape[1] > cfg.inference.max_tokens:
        batch_size = int(cfg.inference.max_tokens / etab.size(1))
    else:
        batch_size = nrgs.shape[1]
    
    # Calculate energies
    scores, scored_seqs, reference_scores = [], [], []
    for batch in range(0, nrgs.shape[1], batch_size):
        batch_scores, batch_seqs, batch_refs = etab_utils.calc_eners(etab, E_idx, seqs[:,batch:batch+batch_size], nrgs[:,batch:batch+batch_size], filter=cfg.inference.filter)
        scores.append(batch_scores)
        scored_seqs.append(batch_seqs)
        reference_scores.append(batch_refs)
    scores, scored_seqs, reference_scores = torch.cat(scores, 1), torch.cat(scored_seqs, 1), torch.cat(reference_scores, 1)

    # Compare to wildtype and remove reference
    scores = scores -scores[:, 0]
    scores, scored_seqs, reference_scores = scores[:, 1:], scored_seqs[:, 1:], reference_scores[:, 1:]

    if cfg.inference.mean_norm: # By default, normalize so mean is 0 (helps when comparing proteins with large numbers of mutants)
        scores -= torch.mean(scores, dim=1)
    return scores, scored_seqs, reference_scores

def score_seqs_singlesite(logprobs, wt_seq, cfg, nrgs, seqs, partition=None):
    """
    Score sequences using the energy table.

    Parameters
    ----------  
    logprobs : torch.Tensor, shape (1, L, vocab)
        Log probabilities
    wt_seq : str, shape (L,)
        Wildtype sequence
    model : PottsMPNN model
        Model with which to score sequences
    cfg : omegaconf
        Config object
    pdb_data : dict
        dict with PDB information
    nrgs : list of shape (N,)
        Mutant energy information
    seqs : list of shape (N, L)
        Mutant sequence information
    partition : list (optional, default None)
        list of chains to analyze
    
    Returns
    -------
    scores : torch.Tensor, shape (N,)
        Scores for each sequence.

    """
    
    # Run energy prediction according to config
    # Use wildtype as reference energy
    wt_tensor = etab_utils.seq_to_tensor(wt_seq).to(dtype=torch.int64, device=cfg.dev).unsqueeze(0).unsqueeze(-1)  # Shape (1, 1, L)
    wt_logprob_vals = logprobs.gather(2, wt_tensor).squeeze(0).squeeze(-1)  # Shape (L,)
    wt_ener = torch.sum(wt_logprob_vals)
    scores = []
    for mut_seq, mut_ener in zip(seqs, nrgs):
        try:
            seq_tensor = etab_utils.seq_to_tensor(mut_seq).to(dtype=torch.int64, device=cfg.dev).unsqueeze(0).unsqueeze(-1)  # Shape (1, 1, L)
            logprob_vals = logprobs.gather(2, seq_tensor).squeeze(0).squeeze(-1)  # Shape (L,)
            mut_ener = torch.sum(logprob_vals)
            mut_ener = mut_ener - wt_ener
            scores.append(mut_ener)
        except Exception as e:
            print(f"Error processing mutant sequence {mut_seq} with energy {mut_ener}: {e}")
            continue
    scores = torch.stack(scores)  # Shape (N,)

    return scores


def plot_data(data,
              only_mutated_positions=False,
              title='PottsMPNN Predictions',
              clabel=r'Predicted $\Delta\Delta$G (a.u.)',
              save_path=None,
              figsize=(20, 5),
              ener_type='ddG',
              chain_ranges=None,
              chain_order=None):
    """
    Plots a heatmap of mutation energies from a dataframe.

    Parameters:
    - data : DataFrame with columns 'mutant', 'wildtype', 'ddG_pred'.
            Sequences use ':' as chain delimiters.
    - only_mutated_positions : If True, only plots columns (residues) that have at least one mutation.
    - chain_ranges : Dict { 'A': [start, stop] } defining inclusive 1-indexed ranges for specific chains.
    - chain_order : List of strings (e.g. ['H', 'L']). 
                   1. Maps the split input sequences to these names (Index 0 -> chain_order[0]).
                   2. Determines the order in which chains are plotted.
                   If None, defaults to ['A', 'B', 'C'...] and alphabetical sort.
    """

    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}

    # --- 1. Parse Data ---
    parsed_data = {}     # parsed_data[chain_name][pos] = {wt, muts}
    chain_sequences = {} # chain_sequences[chain_name] = list(sequence)

    for _, row in data.iterrows():
        wt_seq = row['wildtype']
        mut_seq = row['mutant']
        energy = row['ddG_pred']

        wt_chains = wt_seq.split(':')
        mut_chains = mut_seq.split(':')

        if len(wt_chains) != len(mut_chains):
            continue 

        # Determine Chain Names for this row
        current_chain_names = []
        if chain_order:
            current_chain_names = chain_order[:len(wt_chains)]
        else:
            current_chain_names = [chr(65 + i) for i in range(len(wt_chains))]

        # Identify mutations
        global_mutations = [] 
        
        for c_name, w_chain, m_chain in zip(current_chain_names, wt_chains, mut_chains):
            if len(w_chain) != len(m_chain): continue 
            
            # Store WT sequence logic (first time we see this chain name)
            if c_name not in chain_sequences:
                chain_sequences[c_name] = list(w_chain)
            
            # Find mismatches
            for i, (w, m) in enumerate(zip(w_chain, m_chain)):
                if w != m:
                    # 1-indexed position
                    global_mutations.append((c_name, i + 1, w, m))

        # Constraint: Only single mutations allowed per row
        if len(global_mutations) == 1:
            c_name, pos, wt, mut = global_mutations[0]
            
            if c_name not in parsed_data: parsed_data[c_name] = {}
            if pos not in parsed_data[c_name]: parsed_data[c_name][pos] = {'wt': wt, 'muts': {}}
            
            parsed_data[c_name][pos]['muts'][mut] = energy

    # --- 2. Determine Chains to Plot ---
    if chain_order:
        active_chain_names = [c for c in chain_order if c in chain_sequences]
    else:
        active_chain_names = sorted(chain_sequences.keys())

    if not active_chain_names:
        print("No valid data found to plot.")
        return

    # --- 3. Construct Matrix Columns ---
    matrix_columns = []   # List of (chain_name, pos, wt_residue)
    chain_boundaries = [] # List of column indices where new chains start

    current_col_idx = 0
    for c_name in active_chain_names:
        chain_boundaries.append(current_col_idx)
        full_seq = chain_sequences[c_name]
        
        # Determine valid range for this chain
        start_r, stop_r = 1, len(full_seq)
        if chain_ranges and c_name in chain_ranges:
            start_r, stop_r = chain_ranges[c_name]
            if start_r == 0:
                start_r = 1
            if stop_r == -1:
                stop_r = len(full_seq)
        elif chain_ranges:
            continue 

        # Determine which positions to include
        if only_mutated_positions:
            existing_pos = sorted(parsed_data.get(c_name, {}).keys())
            positions = [p for p in existing_pos if start_r <= p <= stop_r]
        else:
            actual_start = max(1, start_r)
            actual_stop = min(len(full_seq), stop_r)
            if actual_start > actual_stop:
                positions = []
            else:
                positions = range(actual_start, actual_stop + 1)

        for pos in positions:
            wt_aa = full_seq[pos - 1] # 0-indexed lookup
            matrix_columns.append((c_name, pos, wt_aa))
            current_col_idx += 1
            
    # Initialize matrix
    heatmap_data = np.full((len(amino_acids), len(matrix_columns)), np.nan)

    # Fill matrix
    for col_idx, (c_name, pos, wt_aa) in enumerate(matrix_columns):
        if ener_type == 'ddG' and wt_aa in aa_to_idx:
            heatmap_data[aa_to_idx[wt_aa], col_idx] = 0.0

        if c_name in parsed_data and pos in parsed_data[c_name]:
            muts = parsed_data[c_name][pos]['muts']
            for mut_aa, ener in muts.items():
                if mut_aa in aa_to_idx:
                    row_idx = aa_to_idx[mut_aa]
                    heatmap_data[row_idx, col_idx] = ener

    # --- 4. Plotting ---
    blue = (0.0, 0.0, 1.0)
    gray90 = (0.9, 0.9, 0.9)
    red = (1.0, 0.0, 0.0)
    cmap = mcolors.LinearSegmentedColormap.from_list("Blue_Gray90_Red", [blue, gray90, red])
    
    if ener_type == 'ddG':
        center = 0
    else:
        center = np.nanmean(heatmap_data)

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    sns.set(font_scale=0.8)
    ax.set_facecolor('#E0E0E0')

    # Prepare labels
    tick_labels = [f"{wt}{pos}" for (_, pos, wt) in matrix_columns]

    sns.heatmap(
        heatmap_data,
        cmap=cmap,
        center=center,
        yticklabels=amino_acids,
        xticklabels=False, 
        cbar_kws={'shrink': 0.8, 'pad': 0.02, 'label': clabel},
        mask=np.isnan(heatmap_data),
        ax=ax
    )
    ax.collections[0].colorbar.ax.set_ylabel(clabel, fontsize=12) 
    ax.collections[0].colorbar.ax.tick_params(labelsize=12)

    # --- 5. Styling Missing Data (Exact 'X' using Lines) ---
    segments = []
    rows, cols = heatmap_data.shape
    for r in range(rows):
        for c in range(cols):
            if np.isnan(heatmap_data[r, c]):
                p1 = (c, r)
                p2 = (c + 1, r + 1)
                p3 = (c, r + 1)
                p4 = (c + 1, r)
                segments.append([p1, p2])
                segments.append([p3, p4])
    
    if segments:
        lc = LineCollection(segments, color='gray', linewidths=0.5, alpha=0.5)
        ax.add_collection(lc)

    # --- 6. Formatting Axes & Borders ---
    for tick in ax.get_yticklabels():
        tick.set_rotation(0)
        tick.set_ha('left')
        tick.set_position((-0.02, tick.get_position()[1]))
        tick.set_fontsize(12)

    # Font size calculation
    n_cols = len(matrix_columns)
    tick_indices = np.arange(0, n_cols, 1)
    tick_locs = tick_indices + 0.5
    fig_w, _ = fig.get_size_inches()
    ax_w_frac = ax.get_position().width
    box_w_inches = (fig_w * ax_w_frac) / max(1, n_cols)
    max_font_size = box_w_inches * 72 * 0.9
    final_fontsize = min(12, max_font_size)

    ax.set_xticks(tick_locs)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=final_fontsize)

    plt.xlabel('Wildtype Residue', fontsize=12, labelpad=25) 
    plt.ylabel('Mutant Residue', fontsize=12)
    plt.title(title, fontsize=12)

    # Add Borders around Chains & Chain Labels
    boundaries = chain_boundaries + [len(matrix_columns)]
    
    for i, c_name in enumerate(active_chain_names):
        if chain_ranges and c_name not in chain_ranges:
            continue
        start = boundaries[i]
        end = boundaries[i+1]
        width = end - start
        height = len(amino_acids)
        
        # 1. Draw Border
        rect = Rectangle((start, 0), width, height, 
                         fill=False, edgecolor='black', lw=2, clip_on=False)
        ax.add_patch(rect)

        # 2. Add Chain Label
        # Calculate center in data coordinates (x-axis)
        center_x = (start + end) / 2
        
        ax.text(center_x, -0.14, f"Chain {c_name}", 
                ha='center', va='top', fontsize=12, fontweight='bold',
                transform=ax.get_xaxis_transform())

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()