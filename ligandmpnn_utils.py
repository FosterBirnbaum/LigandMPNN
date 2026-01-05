import pandas as pd
import pickle
import webdataset as wds
import os
import torch
import numpy as np
from data_utils import parse_PDB, get_nearest_neighbours, alphabet
from model_utils import nlcpl

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