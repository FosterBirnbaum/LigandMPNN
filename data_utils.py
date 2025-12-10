from __future__ import print_function

import numpy as np
import torch
import torch.utils
from typing import Optional
from dataclasses import dataclass
import time
import os
import random
import math
from torch.utils.data import Sampler
from prody import *
from torch.nn.utils.rnn import pad_sequence
confProDy(verbosity="none")

restype_1to3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
    "X": "UNK",
}
restype_str_to_int = {
    "A": 0,
    "C": 1,
    "D": 2,
    "E": 3,
    "F": 4,
    "G": 5,
    "H": 6,
    "I": 7,
    "K": 8,
    "L": 9,
    "M": 10,
    "N": 11,
    "P": 12,
    "Q": 13,
    "R": 14,
    "S": 15,
    "T": 16,
    "V": 17,
    "W": 18,
    "Y": 19,
    "X": 20,
}
restype_int_to_str = {
    0: "A",
    1: "C",
    2: "D",
    3: "E",
    4: "F",
    5: "G",
    6: "H",
    7: "I",
    8: "K",
    9: "L",
    10: "M",
    11: "N",
    12: "P",
    13: "Q",
    14: "R",
    15: "S",
    16: "T",
    17: "V",
    18: "W",
    19: "Y",
    20: "X",
}
alphabet = list(restype_str_to_int)

element_list = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mb",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Uut",
    "Fl",
    "Uup",
    "Lv",
    "Uus",
    "Uuo",
]
element_list = [item.upper() for item in element_list]
# element_dict = dict(zip(element_list, range(1,len(element_list))))
element_dict_rev = dict(zip(range(1, len(element_list)), element_list))


def get_seq_rec(S: torch.Tensor, S_pred: torch.Tensor, mask: torch.Tensor):
    """
    S : true sequence shape=[batch, length]
    S_pred : predicted sequence shape=[batch, length]
    mask : mask to compute average over the region shape=[batch, length]

    average : averaged sequence recovery shape=[batch]
    """
    match = S == S_pred
    average = torch.sum(match * mask, dim=-1) / torch.sum(mask, dim=-1)
    return average


def get_score(S: torch.Tensor, log_probs: torch.Tensor, mask: torch.Tensor):
    """
    S : true sequence shape=[batch, length]
    log_probs : predicted sequence shape=[batch, length]
    mask : mask to compute average over the region shape=[batch, length]

    average_loss : averaged categorical cross entropy (CCE) [batch]
    loss_per_resdue : per position CCE [batch, length]
    """
    S_one_hot = torch.nn.functional.one_hot(S, 21)
    loss_per_residue = -(S_one_hot * log_probs).sum(-1)  # [B, L]
    average_loss = torch.sum(loss_per_residue * mask, dim=-1) / (
        torch.sum(mask, dim=-1) + 1e-8
    )
    return average_loss, loss_per_residue


def write_full_PDB(
    save_path: str,
    X: np.ndarray,
    X_m: np.ndarray,
    b_factors: np.ndarray,
    R_idx: np.ndarray,
    chain_letters: np.ndarray,
    S: np.ndarray,
    other_atoms=None,
    icodes=None,
    force_hetatm=False,
):
    """
    save_path : path where the PDB will be written to
    X : protein atom xyz coordinates shape=[length, 14, 3]
    X_m : protein atom mask shape=[length, 14]
    b_factors: shape=[length, 14]
    R_idx: protein residue indices shape=[length]
    chain_letters: protein chain letters shape=[length]
    S : protein amino acid sequence shape=[length]
    other_atoms: other atoms parsed by prody
    icodes: a list of insertion codes for the PDB; e.g. antibody loops
    """

    restype_1to3 = {
        "A": "ALA",
        "R": "ARG",
        "N": "ASN",
        "D": "ASP",
        "C": "CYS",
        "Q": "GLN",
        "E": "GLU",
        "G": "GLY",
        "H": "HIS",
        "I": "ILE",
        "L": "LEU",
        "K": "LYS",
        "M": "MET",
        "F": "PHE",
        "P": "PRO",
        "S": "SER",
        "T": "THR",
        "W": "TRP",
        "Y": "TYR",
        "V": "VAL",
        "X": "UNK",
    }
    restype_INTtoSTR = {
        0: "A",
        1: "C",
        2: "D",
        3: "E",
        4: "F",
        5: "G",
        6: "H",
        7: "I",
        8: "K",
        9: "L",
        10: "M",
        11: "N",
        12: "P",
        13: "Q",
        14: "R",
        15: "S",
        16: "T",
        17: "V",
        18: "W",
        19: "Y",
        20: "X",
    }
    restype_name_to_atom14_names = {
        "ALA": ["N", "CA", "C", "O", "CB", "", "", "", "", "", "", "", "", ""],
        "ARG": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "CD",
            "NE",
            "CZ",
            "NH1",
            "NH2",
            "",
            "",
            "",
        ],
        "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2", "", "", "", "", "", ""],
        "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2", "", "", "", "", "", ""],
        "CYS": ["N", "CA", "C", "O", "CB", "SG", "", "", "", "", "", "", "", ""],
        "GLN": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "CD",
            "OE1",
            "NE2",
            "",
            "",
            "",
            "",
            "",
        ],
        "GLU": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "CD",
            "OE1",
            "OE2",
            "",
            "",
            "",
            "",
            "",
        ],
        "GLY": ["N", "CA", "C", "O", "", "", "", "", "", "", "", "", "", ""],
        "HIS": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "ND1",
            "CD2",
            "CE1",
            "NE2",
            "",
            "",
            "",
            "",
        ],
        "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", "", "", "", "", "", ""],
        "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "", "", "", "", "", ""],
        "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "", "", "", "", ""],
        "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE", "", "", "", "", "", ""],
        "PHE": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "CD1",
            "CD2",
            "CE1",
            "CE2",
            "CZ",
            "",
            "",
            "",
        ],
        "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD", "", "", "", "", "", "", ""],
        "SER": ["N", "CA", "C", "O", "CB", "OG", "", "", "", "", "", "", "", ""],
        "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2", "", "", "", "", "", "", ""],
        "TRP": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "CD1",
            "CD2",
            "CE2",
            "CE3",
            "NE1",
            "CZ2",
            "CZ3",
            "CH2",
        ],
        "TYR": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "CD1",
            "CD2",
            "CE1",
            "CE2",
            "CZ",
            "OH",
            "",
            "",
        ],
        "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "", "", "", "", "", "", ""],
        "UNK": ["", "", "", "", "", "", "", "", "", "", "", "", "", ""],
    }

    S_str = [restype_1to3[AA] for AA in [restype_INTtoSTR[AA] for AA in S]]

    X_list = []
    b_factor_list = []
    atom_name_list = []
    element_name_list = []
    residue_name_list = []
    residue_number_list = []
    chain_id_list = []
    icodes_list = []
    for i, AA in enumerate(S_str):
        sel = X_m[i].astype(np.int32) == 1
        total = np.sum(sel)
        tmp = np.array(restype_name_to_atom14_names[AA])[sel]
        X_list.append(X[i][sel])
        b_factor_list.append(b_factors[i][sel])
        atom_name_list.append(tmp)
        element_name_list += [AA[:1] for AA in list(tmp)]
        residue_name_list += total * [AA]
        residue_number_list += total * [R_idx[i]]
        chain_id_list += total * [chain_letters[i]]
        icodes_list += total * [icodes[i]]

    X_stack = np.concatenate(X_list, 0)
    b_factor_stack = np.concatenate(b_factor_list, 0)
    atom_name_stack = np.concatenate(atom_name_list, 0)

    protein = prody.AtomGroup()
    protein.setCoords(X_stack)
    protein.setBetas(b_factor_stack)
    protein.setNames(atom_name_stack)
    protein.setResnames(residue_name_list)
    protein.setElements(element_name_list)
    protein.setOccupancies(np.ones([X_stack.shape[0]]))
    protein.setResnums(residue_number_list)
    protein.setChids(chain_id_list)
    protein.setIcodes(icodes_list)

    if other_atoms:
        other_atoms_g = prody.AtomGroup()
        other_atoms_g.setCoords(other_atoms.getCoords())
        other_atoms_g.setNames(other_atoms.getNames())
        other_atoms_g.setResnames(other_atoms.getResnames())
        other_atoms_g.setElements(other_atoms.getElements())
        other_atoms_g.setOccupancies(other_atoms.getOccupancies())
        other_atoms_g.setResnums(other_atoms.getResnums())
        other_atoms_g.setChids(other_atoms.getChids())
        if force_hetatm:
            other_atoms_g.setFlags("hetatm", other_atoms.getFlags("hetatm"))
        writePDB(save_path, protein + other_atoms_g)
    else:
        writePDB(save_path, protein)


def get_aligned_coordinates(protein_atoms, CA_dict: dict, atom_name: str):
    """
    protein_atoms: prody atom group
    CA_dict: mapping between chain_residue_idx_icodes and integers
    atom_name: atom to be parsed; e.g. CA
    """
    atom_atoms = protein_atoms.select(f"name {atom_name}")

    if atom_atoms != None:
        atom_coords = atom_atoms.getCoords()
        atom_resnums = atom_atoms.getResnums()
        atom_chain_ids = atom_atoms.getChids()
        atom_icodes = atom_atoms.getIcodes()

    atom_coords_ = np.zeros([len(CA_dict), 3], np.float32)
    atom_coords_m = np.zeros([len(CA_dict)], np.int32)
    if atom_atoms != None:
        for i in range(len(atom_resnums)):
            code = atom_chain_ids[i] + "_" + str(atom_resnums[i]) + "_" + atom_icodes[i]
            if code in list(CA_dict):
                atom_coords_[CA_dict[code], :] = atom_coords[i]
                atom_coords_m[CA_dict[code]] = 1
    return atom_coords_, atom_coords_m


def parse_PDB(
    input_path: str,
    device: str = "cpu",
    chains: list = [],
    segids: list = [],
    parse_all_atoms: bool = False,
    parse_atoms_with_zero_occupancy: bool = False,
    masked_list: list = [],
    epoch: int = 0,
    updated_alist: bool = False,
    modify_list: list = [],
):
    """
    input_path : path for the input PDB
    device: device for the torch.Tensor
    chains: a list specifying which chains need to be parsed; e.g. ["A", "B"]
    parse_all_atoms: if False parse only N,CA,C,O otherwise all 37 atoms
    parse_atoms_with_zero_occupancy: if True atoms with zero occupancy will be parsed
    """
    element_list = [
        "H",
        "He",
        "Li",
        "Be",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Ne",
        "Na",
        "Mg",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "Ar",
        "K",
        "Ca",
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Ga",
        "Ge",
        "As",
        "Se",
        "Br",
        "Kr",
        "Rb",
        "Sr",
        "Y",
        "Zr",
        "Nb",
        "Mb",
        "Tc",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "In",
        "Sn",
        "Sb",
        "Te",
        "I",
        "Xe",
        "Cs",
        "Ba",
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
        "Hf",
        "Ta",
        "W",
        "Re",
        "Os",
        "Ir",
        "Pt",
        "Au",
        "Hg",
        "Tl",
        "Pb",
        "Bi",
        "Po",
        "At",
        "Rn",
        "Fr",
        "Ra",
        "Ac",
        "Th",
        "Pa",
        "U",
        "Np",
        "Pu",
        "Am",
        "Cm",
        "Bk",
        "Cf",
        "Es",
        "Fm",
        "Md",
        "No",
        "Lr",
        "Rf",
        "Db",
        "Sg",
        "Bh",
        "Hs",
        "Mt",
        "Ds",
        "Rg",
        "Cn",
        "Uut",
        "Fl",
        "Uup",
        "Lv",
        "Uus",
        "Uuo",
    ]
    if updated_alist:
        element_list += ["Unx", "X"]

    
    element_list = [item.upper() for item in element_list]
    element_dict = dict(zip(element_list, range(1, len(element_list) + 1)))
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
    restype_STRtoINT = {
        "A": 0,
        "C": 1,
        "D": 2,
        "E": 3,
        "F": 4,
        "G": 5,
        "H": 6,
        "I": 7,
        "K": 8,
        "L": 9,
        "M": 10,
        "N": 11,
        "P": 12,
        "Q": 13,
        "R": 14,
        "S": 15,
        "T": 16,
        "V": 17,
        "W": 18,
        "Y": 19,
        "X": 20,
    }

    atom_order = {
        "N": 0,
        "CA": 1,
        "C": 2,
        "CB": 3,
        "O": 4,
        "CG": 5,
        "CG1": 6,
        "CG2": 7,
        "OG": 8,
        "OG1": 9,
        "SG": 10,
        "CD": 11,
        "CD1": 12,
        "CD2": 13,
        "ND1": 14,
        "ND2": 15,
        "OD1": 16,
        "OD2": 17,
        "SD": 18,
        "CE": 19,
        "CE1": 20,
        "CE2": 21,
        "CE3": 22,
        "NE": 23,
        "NE1": 24,
        "NE2": 25,
        "OE1": 26,
        "OE2": 27,
        "CH2": 28,
        "NH1": 29,
        "NH2": 30,
        "OH": 31,
        "CZ": 32,
        "CZ2": 33,
        "CZ3": 34,
        "NZ": 35,
        "OXT": 36,
    }

    if not parse_all_atoms:
        atom_types = ["N", "CA", "C", "O"]
    else:
        atom_types = [
            "N",
            "CA",
            "C",
            "CB",
            "O",
            "CG",
            "CG1",
            "CG2",
            "OG",
            "OG1",
            "SG",
            "CD",
            "CD1",
            "CD2",
            "ND1",
            "ND2",
            "OD1",
            "OD2",
            "SD",
            "CE",
            "CE1",
            "CE2",
            "CE3",
            "NE",
            "NE1",
            "NE2",
            "OE1",
            "OE2",
            "CH2",
            "NH1",
            "NH2",
            "OH",
            "CZ",
            "CZ2",
            "CZ3",
            "NZ",
        ]

    atoms = parsePDB(input_path)

    if not parse_atoms_with_zero_occupancy:
        atoms = atoms.select("occupancy > 0")
    if chains:
        str_out = ""
        for item in chains:
            str_out += " chain " + item + " or"
        atoms = atoms.select(str_out[1:-3])
    elif segids:
        str_out = ""
        for item in chains:
            str_out += " segment " + item + " or"
        atoms = atoms.select(str_out[1:-3])
    try:
        protein_atoms = atoms.select("protein")
        backbone = protein_atoms.select("backbone")
    except Exception as e:
        # with open('/orcd/scratch/orcd/001/fosterb/pmpnn_experiments/ligandmpnn_potts_nomixed/log_error.out', 'a') as f:
        #     f.write(str(input_path) + '\n')
        #     f.write(str(e) + '\n')
        return None, None, None, None, None
    other_atoms = atoms.select("not protein and not water")
    water_atoms = atoms.select("water")

    CA_atoms = protein_atoms.select("name CA")

    if os.path.basename(input_path.split('.')[0]) in modify_list:
        CA_atoms = CA_atoms[:-2]
    CA_resnums = CA_atoms.getResnums()
    CA_chain_ids = CA_atoms.getChids()
    CA_icodes = CA_atoms.getIcodes()

    CA_dict = {}
    for i in range(len(CA_resnums)):
        code = CA_chain_ids[i] + "_" + str(CA_resnums[i]) + "_" + CA_icodes[i]
        CA_dict[code] = i

    xyz_37 = np.zeros([len(CA_dict), 37, 3], np.float32)
    xyz_37_m = np.zeros([len(CA_dict), 37], np.int32)
    for atom_name in atom_types:
        xyz, xyz_m = get_aligned_coordinates(protein_atoms, CA_dict, atom_name)
        xyz_37[:, atom_order[atom_name], :] = xyz
        xyz_37_m[:, atom_order[atom_name]] = xyz_m

    N = xyz_37[:, atom_order["N"], :]
    CA = xyz_37[:, atom_order["CA"], :]
    C = xyz_37[:, atom_order["C"], :]
    O = xyz_37[:, atom_order["O"], :]

    N_m = xyz_37_m[:, atom_order["N"]]
    CA_m = xyz_37_m[:, atom_order["CA"]]
    C_m = xyz_37_m[:, atom_order["C"]]
    O_m = xyz_37_m[:, atom_order["O"]]

    mask = N_m * CA_m * C_m * O_m  # must all 4 atoms exist

    b = CA - N
    c = C - CA
    a = np.cross(b, c, axis=-1)
    CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA

    chain_labels = np.array(CA_atoms.getChindices(), dtype=np.int32)
    R_idx = np.array(CA_resnums, dtype=np.int32)
    S = CA_atoms.getResnames()
    S = [restype_3to1[AA] if AA in list(restype_3to1) else "X" for AA in list(S)]
    S = np.array([restype_STRtoINT[AA] for AA in list(S)], np.int32)
    X = np.concatenate([N[:, None], CA[:, None], C[:, None], O[:, None]], 1)

    try:
        Y = np.array(other_atoms.getCoords(), dtype=np.float32)
        Y_t = list(other_atoms.getElements())
        Y_t = np.array(
            [
                element_dict[y_t.upper()] if y_t.upper() in element_list else 0
                for y_t in Y_t
            ],
            dtype=np.int32,
        )
        Y_m = (Y_t != 1) * (Y_t != 0)

        Y = Y[Y_m, :]
        Y_t = Y_t[Y_m]
        Y_m = Y_m[Y_m]
    except Exception as e:
        # if epoch == 0:
        #     with open('/orcd/scratch/orcd/001/fosterb/pmpnn_experiments/ligandmpnn_potts_nomixed/data_err.txt', 'a') as f:
        #         f.write(str(input_path) + '\n')
        #         f.write(str(e) + '\n')
        # return None, None, None, None, None
        Y = np.zeros([1, 3], np.float32)
        Y_t = np.zeros([1], np.int32)
        Y_m = np.zeros([1], np.int32)
    if Y.shape[0] == 0:
        # with open('/orcd/scratch/orcd/001/fosterb/pmpnn_experiments/ligandmpnn_potts_nomixed/data_err.txt', 'a') as f:
        #     f.write(str(input_path) + '\n')
        Y = np.zeros([1, 3], np.float32)
        Y_t = np.zeros([1], np.int32)
        Y_m = np.zeros([1], np.int32)

    output_dict = {}
    output_dict["X"] = torch.tensor(X, device=device, dtype=torch.float32)
    output_dict["mask"] = torch.tensor(mask, device=device, dtype=torch.int32)
    output_dict["Y"] = torch.tensor(Y, device=device, dtype=torch.float32)
    output_dict["Y_t"] = torch.tensor(Y_t, device=device, dtype=torch.int32)
    output_dict["Y_m"] = torch.tensor(Y_m, device=device, dtype=torch.int32)

    output_dict["R_idx"] = torch.tensor(R_idx, device=device, dtype=torch.int32)
    output_dict["chain_labels"] = torch.tensor(
        chain_labels, device=device, dtype=torch.int32
    )

    output_dict["chain_letters"] = CA_chain_ids

    mask_c = []
    chain_list = list(set(output_dict["chain_letters"]))
    chain_list.sort()
    for chain in chain_list:
        mask_c.append(
            torch.tensor(
                [chain == item for item in output_dict["chain_letters"]],
                device=device,
                dtype=bool,
            )
        )

    output_dict["mask_c"] = mask_c
    output_dict["chain_list"] = chain_list

    output_dict["S"] = torch.tensor(S, device=device, dtype=torch.int32)

    output_dict["xyz_37"] = torch.tensor(xyz_37, device=device, dtype=torch.float32)
    output_dict["xyz_37_m"] = torch.tensor(xyz_37_m, device=device, dtype=torch.int32)
    output_dict['chain_mask'] = torch.tensor(
            np.array(
                [
                    item not in masked_list
                    for item in output_dict["chain_letters"]
                ],
                dtype=np.int32,
            ),
            device=device,
        )

    return output_dict, backbone, other_atoms, CA_icodes, water_atoms


def get_nearest_neighbours(CB, mask, Y, Y_t, Y_m, number_of_ligand_atoms):
    device = CB.device
    mask_CBY = mask[:, None] * Y_m[None, :]  # [A,B]
    L2_AB = torch.sum((CB[:, None, :] - Y[None, :, :]) ** 2, -1)
    L2_AB = L2_AB * mask_CBY + (1 - mask_CBY) * 1000.0

    nn_idx = torch.argsort(L2_AB, -1)[:, :number_of_ligand_atoms]
    L2_AB_nn = torch.gather(L2_AB, 1, nn_idx)
    D_AB_closest = torch.sqrt(L2_AB_nn[:, 0])

    Y_r = Y[None, :, :].repeat(CB.shape[0], 1, 1)
    Y_t_r = Y_t[None, :].repeat(CB.shape[0], 1)
    Y_m_r = Y_m[None, :].repeat(CB.shape[0], 1)

    Y_tmp = torch.gather(Y_r, 1, nn_idx[:, :, None].repeat(1, 1, 3))
    Y_t_tmp = torch.gather(Y_t_r, 1, nn_idx)
    Y_m_tmp = torch.gather(Y_m_r, 1, nn_idx)

    Y = torch.zeros(
        [CB.shape[0], number_of_ligand_atoms, 3], dtype=torch.float32, device=device
    )
    Y_t = torch.zeros(
        [CB.shape[0], number_of_ligand_atoms], dtype=torch.int32, device=device
    )
    Y_m = torch.zeros(
        [CB.shape[0], number_of_ligand_atoms], dtype=torch.int32, device=device
    )

    num_nn_update = Y_tmp.shape[1]
    Y[:, :num_nn_update] = Y_tmp
    Y_t[:, :num_nn_update] = Y_t_tmp
    Y_m[:, :num_nn_update] = Y_m_tmp

    return Y, Y_t, Y_m, D_AB_closest

def _to_dev(data_dict, dev, rank=None):
    """ Push all tensor objects in the dictionary to the given device.

    Args
    ----
    data_dict : dict
        Dictionary of input features to TERMinator
    dev : str
        Device to load tensors onto
    """
    for key, value in data_dict.items():
        if isinstance(value, torch.Tensor):
            data_dict[key] = value.to(dev)
            if rank is not None:
                data_dict[key] = value.to(rank)
            if data_dict[key].is_floating_point():
                data_dict[key].requires_grad = True

def featurize(
    input_dicts,
    cutoff_for_score=8.0,
    use_atom_context=True,
    number_of_ligand_atoms=16,
    model_type="protein_mpnn",
    device="cpu",
    ligand_mpnn_use_side_chain_context=0,
    parse_atoms_with_zero_occupancy=1,
    transmembrane_buried="",
    transmembrane_interface="",
    global_transmembrane_label=0,
    subset_chains=False,
    epoch=0,
    updated_alist=True
):
   
    # try:
    output_dict_list = []
    batched_output_dict = {}
    use_side_chain_mask = False
    use_xyz_37 = False
    max_cl = 0

    for input_dict in input_dicts:
        output_dict = {'filepath': input_dict['filepath']}

        # make protein dict
        if subset_chains:
            segids = input_dict['chains']
        else:
            segids = []
        input_dict, _, _, icodes, _ = parse_PDB(
                        input_dict['filepath'],
                        device='cpu',
                        segids=segids,
                        parse_all_atoms=ligand_mpnn_use_side_chain_context,
                        parse_atoms_with_zero_occupancy=parse_atoms_with_zero_occupancy, ## FIX INPUT MASKED LIST
                        epoch=epoch,
                        updated_alist=updated_alist
                    )
        # try:
            
        # except Exception as e:
        #     input_dict_str = str(input_dict)
        #     with open('/orcd/scratch/orcd/001/fosterb/pmpnn_experiments/ligandmpnn_potts_nomixed/test.out', 'w') as f:
        #         f.write(input_dict_str + '\n')
        #         f.write(str(e) + '\n')
        #     raise ValueError
        #     continue
        if input_dict is None:
            # raise ValueError
            continue

        # make chain_letter + residue_idx + insertion_code mapping to integers
        R_idx_list = list(input_dict["R_idx"].cpu().numpy())  # residue indices
        chain_letters_list = list(input_dict["chain_letters"])  # chain letters
        encoded_residues = []
        for i, R_idx_item in enumerate(R_idx_list):
            tmp = str(chain_letters_list[i]) + str(R_idx_item) + icodes[i]
            encoded_residues.append(tmp)

        fixed_positions = torch.tensor(
            [int(True) for item in encoded_residues],
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
        input_dict["membrane_per_residue_labels"] = 2 * buried_positions * (
            1 - interface_positions
        ) + 1 * interface_positions * (1 - buried_positions)

        if model_type == "global_label_membrane_mpnn":
            input_dict["membrane_per_residue_labels"] = (
                global_transmembrane_label + 0 * fixed_positions
            )

        # create chain_mask
        chains_to_design_list = input_dict["chain_letters"]
        chain_mask = torch.tensor(
            np.array(
                [
                    item in chains_to_design_list
                    for item in input_dict["chain_letters"]
                ],
                dtype=np.int32,
            ),
            device=device,
        )
        input_dict["chain_mask"] = chain_mask


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
            try:
                Y, Y_t, Y_m, D_XY = get_nearest_neighbours(
                    CB, mask, Y, Y_t, Y_m, number_of_ligand_atoms
                )
            except Exception as e:
                # with open('/orcd/scratch/orcd/001/fosterb/pmpnn_experiments/ligandmpnn_potts_nomixed/test2.out', 'w') as f:
                #     f.write(str(Y.shape) + ' ' + str(Y_t.shape) + ' ' + str(Y_m.shape) + ' ' + str(mask.shape) + ' ' + str(CB.shape) + '\n')
                #     f.write(str(Y) + '\n')
                #     f.write(str(Y_t) + '\n')
                #     f.write(str(Y_m) + '\n')
                #     f.write(str(e) + '\n')
                #     f.write(str(output_dict['filepath']))
                raise e
            mask_XY = (D_XY < cutoff_for_score) * mask * Y_m[:, 0]
            output_dict["mask_XY"] = mask_XY
            if "side_chain_mask" in list(input_dict):
                output_dict["side_chain_mask"] = input_dict["side_chain_mask"]
            output_dict["Y"] = Y
            output_dict["Y_t"] = Y_t
            output_dict["Y_m"] = Y_m
            if not use_atom_context:
                output_dict["Y_m"] = 0.0 * output_dict["Y_m"]
        elif (
            model_type == "per_residue_label_membrane_mpnn"
            or model_type == "global_label_membrane_mpnn"
        ):
            output_dict["membrane_per_residue_labels"] = input_dict[
                "membrane_per_residue_labels"
            ]

        R_idx_list = []
        count = 0
        R_idx_prev = -100000
        for R_idx in list(input_dict["R_idx"]):
            if R_idx_prev == R_idx:
                count += 1
            R_idx_list.append(R_idx + count)
            R_idx_prev = R_idx
        R_idx_renumbered = torch.tensor(R_idx_list, device=R_idx.device)
        output_dict["R_idx"] = R_idx_renumbered
        output_dict["R_idx_original"] = input_dict["R_idx"]
        output_dict["chain_labels"] = input_dict["chain_labels"]
        output_dict["S"] = input_dict["S"]
        output_dict["chain_mask"] = input_dict["chain_mask"]
        output_dict["mask"] = input_dict["mask"]
        output_dict["X"] = input_dict["X"]

        output_dict["randn"] =  torch.randn(
            [output_dict["mask"].shape[0]],
            device=device,
    )

        if "xyz_37" in list(input_dict):
            output_dict["xyz_37"] = input_dict["xyz_37"]
            output_dict["xyz_37_m"] = input_dict["xyz_37_m"]
            use_xyz_37 = True

        max_cl = max(max_cl, output_dict["X"].shape[1])
        output_dict_list.append(output_dict)
        if "side_chain_mask" in list(input_dict):
            use_side_chain_mask = True
        
    if len(output_dict_list) == 0:
        return None

    # with open('/orcd/scratch/orcd/001/fosterb/pmpnn_experiments/ligandmpnn_potts_nomixed/log_feat.out', 'a') as f:
    #     for out_dict in output_dict_list:
    #         f.write('\t' + str(out_dict['filepath']) + ' ' + str(out_dict['Y'].shape) + ' ' + str(out_dict['mask'].shape) + ' ' + str(out_dict['X'].shape) + '\n')

    batched_output_dict['filepaths'] = [out_dict['filepath'] for out_dict in output_dict_list]
    batched_output_dict["R_idx"] = pad_sequence( [out_dict["R_idx"] for out_dict in output_dict_list], batch_first=True, padding_value=-1 )
    batched_output_dict["R_idx_original"] = pad_sequence( [out_dict["R_idx_original"] for out_dict in output_dict_list], batch_first=True, padding_value=-1 )
    batched_output_dict["chain_labels"] = pad_sequence( [out_dict["chain_labels"] for out_dict in output_dict_list], batch_first=True, padding_value=-1 )
    batched_output_dict["S"] = pad_sequence( [out_dict["S"] for out_dict in output_dict_list], batch_first=True, padding_value=0 )
    batched_output_dict["chain_mask"] = pad_sequence( [out_dict["chain_mask"] for out_dict in output_dict_list], batch_first=True, padding_value=0 )
    batched_output_dict["mask"] = pad_sequence( [out_dict["mask"] for out_dict in output_dict_list], batch_first=True, padding_value=0 )
    batched_output_dict["X"] = pad_sequence( [out_dict["X"] for out_dict in output_dict_list], batch_first=True, padding_value=0 )
    if model_type == "ligand_mpnn":
        batched_output_dict["mask_XY"] = pad_sequence( [out_dict["mask_XY"] for out_dict in output_dict_list], batch_first=True, padding_value=0 )
        batched_output_dict["Y"] = pad_sequence( [out_dict["Y"] for out_dict in output_dict_list], batch_first=True, padding_value=0 )
        batched_output_dict["Y_t"] = pad_sequence( [out_dict["Y_t"] for out_dict in output_dict_list], batch_first=True, padding_value=0 )
        batched_output_dict["Y_m"] = pad_sequence( [out_dict["Y_m"] for out_dict in output_dict_list], batch_first=True, padding_value=0 )
    elif (
            model_type == "per_residue_label_membrane_mpnn"
            or model_type == "global_label_membrane_mpnn"
        ):
        batched_output_dict["membrane_per_residue_labels"] = pad_sequence( [out_dict["membrane_per_residue_labels"] for out_dict in output_dict_list], batch_first=True, padding_value=-1 )
    if use_side_chain_mask:
        batched_output_dict["side_chain_mask"] = pad_sequence( [out_dict["side_chain_mask"] for out_dict in output_dict_list], batch_first=True, padding_value=0 )
    if use_xyz_37:
        batched_output_dict["xyz_37"] = pad_sequence( [out_dict["xyz_37"] for out_dict in output_dict_list], batch_first=True, padding_value=0 )
        batched_output_dict["xyz_37_m"] = pad_sequence( [out_dict["xyz_37_m"] for out_dict in output_dict_list], batch_first=True, padding_value=0 )
    batched_output_dict["randn"] = pad_sequence( [out_dict["randn"] for out_dict in output_dict_list], batch_first=True, padding_value=max_cl+1 )

    # with open('/orcd/scratch/orcd/001/fosterb/pmpnn_experiments/ligandmpnn_potts_nomixed/log_feat.out', 'a') as f:
    #     f.write(str(batched_output_dict['Y'].shape) + ' ' + str(batched_output_dict['mask'].shape) + ' ' + str(batched_output_dict['X'].shape) + str(batched_output_dict["S"].shape) + '\n')

    # except Exception as e:
    #     with open('/orcd/scratch/orcd/001/fosterb/pmpnn_experiments/ligandmpnn_potts_nomixed/log_error.out', 'a') as f:
    #         f.write(str(e) + '\n')
    #         for idict in input_dicts:
    #             f.write(idict['filepath'] + '\n')
    #     # print(e)
    #     # for idict in input_dicts:
    #     #     print(idict)
    #     return None
    return batched_output_dict

def get_ssm_mutations(seq):
    ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'

        # make mutation list for SSM run
    mutation_list = []
    for seq_pos in range(len(seq)):
        wtAA = seq[seq_pos]
        # check for missing residues
        if wtAA != '-':
            # add each mutation option
            for mutAA in ALPHABET[:-1]:
                mutation_list.append(wtAA + str(seq_pos) + mutAA)
        else:
            mutation_list.append(None)

    return mutation_list

@dataclass
class Mutation:
    position: int
    wildtype: str
    mutation: str
    ddG: Optional[float] = None
    pdb: Optional[str] = ''


class StructureDataset():
    def __init__(self, pdb_dict_list, verbose=False, truncate=None, max_length=100,
        alphabet='ACDEFGHIKLMNPQRSTVWYX'):
        alphabet_set = set([a for a in alphabet])
        discard_count = {
            'bad_chars': 0,
            'too_long': 0,
            'bad_seq_length': 0
        }

        self.data = []

        start = time.time()
        for i, entry in enumerate(pdb_dict_list):
            entry = {k:v[0] for k,v in entry.items() if len(v) > 0}
            seq = entry['seq']

            bad_chars = set([s for s in seq]).difference(alphabet_set)
            if len(bad_chars) == 0:
                if len(entry['seq']) <= max_length:
                    self.data.append(entry)
                else:
                    discard_count['too_long'] += 1
            else:
                #print(name, bad_chars, entry['seq'])
                discard_count['bad_chars'] += 1

            # Truncate early
            if truncate is not None and len(self.data) == truncate:
                return

            if verbose and (i + 1) % 1000 == 0:
                elapsed = time.time() - start
                print('{} entries ({} loaded) in {:.1f} s'.format(len(self.data), i+1, elapsed))

            #print('Discarded', discard_count)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return [self.data[i] for i in idx]
        return self.data[idx]
    
class StructureSampler(Sampler):
    def __init__(self, dataset, num_replicas, rank, batch_size=100, device='cpu', cutoff_for_score=8.0, ligand_mpnn_use_atom_context=1, atom_context_num=25, model_type='ligand_mpnn',
                 ligand_mpnn_use_side_chain_context=0, parse_atoms_with_zero_occupancy=0, transmembrane_buried="", transmembrane_interface="", global_transmembrane_label=0, subset_chains=False, updated_alist=True):
        self.size = len(dataset)
        self.num_replicas = num_replicas
        self.rank = rank
        self.lengths = [len(dataset[i]['seq']) for i in range(self.size)]
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.cutoff_for_score = cutoff_for_score
        self.ligand_mpnn_use_atom_context = ligand_mpnn_use_atom_context
        self.atom_context_num = atom_context_num
        self.model_type = model_type
        self.ligand_mpnn_use_side_chain_context = ligand_mpnn_use_side_chain_context
        self.parse_atoms_with_zero_occupancy = parse_atoms_with_zero_occupancy
        self.transmembrane_buried = transmembrane_buried
        self.transmembrane_interface = transmembrane_interface
        self.global_transmembrane_label = global_transmembrane_label
        self.subset_chains = subset_chains
        self.epoch = 0
        self.updated_alist = updated_alist

        self._cluster()

    def _set_epoch(self, epoch):
        self.epoch = epoch

    def _cluster(self):
        sorted_ix = np.argsort(self.lengths)

        # Cluster into batches of similar sizes
        clusters, batch = [], []
        batch_max = 0
        for ix in sorted_ix:
            size = self.lengths[ix]
            # with open('/orcd/scratch/orcd/001/fosterb/pmpnn_experiments/ligandmpnn_potts_nomixed/log_size.out', 'a') as f:
            #     f.write(str(size) + ' ' + str(len(batch) + 1) + '\n')
            if size * (len(batch) + 1) <= self.batch_size:
                batch.append(ix)
                batch_max = size
            else:
                clusters.append(batch)
                batch, batch_max = [ix], 0
        if len(batch) > 0 and size * (len(batch) + 1) <= self.batch_size:
            clusters.append(batch)
        self.clusters = clusters

    def __len__(self):
        return len(self.clusters)
    
    def package(self, b_idx):

        # with open('/orcd/scratch/orcd/001/fosterb/pmpnn_experiments/ligandmpnn_potts_nomixed/log_package.out', 'a') as f:
        #         for b in b_idx:
        #             f.write(str(b['filepath']) + ' ' + str(len(b['seq'])) + '\n')

        return featurize(b_idx,
                    cutoff_for_score=self.cutoff_for_score,
                    use_atom_context=self.ligand_mpnn_use_atom_context,
                    number_of_ligand_atoms=self.atom_context_num,
                    model_type=self.model_type,
                    ligand_mpnn_use_side_chain_context=self.ligand_mpnn_use_side_chain_context,
                    parse_atoms_with_zero_occupancy=self.parse_atoms_with_zero_occupancy,
                    transmembrane_buried=self.transmembrane_buried,
                    transmembrane_interface=self.transmembrane_interface,
                    global_transmembrane_label=self.global_transmembrane_label,
                    device=self.device,
                    subset_chains=self.subset_chains,
                    epoch=self.epoch,
                    updated_alist=self.updated_alist)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)

        # self._cluster()
        # with open('/orcd/scratch/orcd/001/fosterb/pmpnn_experiments/ligandmpnn_potts_nomixed/log_rank.out', 'a') as f:
        #     f.write('HERE!\n')
        #     for clust in self.clusters:
        #         f.write(str(self.rank) + ' ' + str(clust) + '\n')
        # rank_clusters = self.clusters[self.rank:self.size:self.num_replicas]
        # with open('/orcd/scratch/orcd/001/fosterb/pmpnn_experiments/ligandmpnn_potts_nomixed/log_rank.out', 'a') as f:
        #     f.write('HERE!\n')
        #     for clust in rank_clusters:
        #         f.write(str(self.rank) + ' ' + str(clust) + '\n')
        # raise ValueError
        for b_idx in self.clusters:
            yield b_idx

class CustomDistributedSampler(Sampler):
    def __init__(self, sampler, num_replicas, rank):
        self.sampler = sampler
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        self.indices = list(self.sampler)
        # self.num_samples = int(math.floor(len(self.indices) / self.num_replicas))
        self.num_samples = len(self.indices)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)

        rank_indices = self.indices[self.rank:self.total_size:self.num_replicas]
        return iter(rank_indices)

    def __len__(self):
        return self.num_samples

    def _set_epoch(self, epoch):
        self.epoch = epoch

    def package(self, b_idx):

        return featurize(b_idx,
                    cutoff_for_score=self.sampler.cutoff_for_score,
                    use_atom_context=self.sampler.ligand_mpnn_use_atom_context,
                    number_of_ligand_atoms=self.sampler.atom_context_num,
                    model_type=self.sampler.model_type,
                    ligand_mpnn_use_side_chain_context=self.sampler.ligand_mpnn_use_side_chain_context,
                    parse_atoms_with_zero_occupancy=self.sampler.parse_atoms_with_zero_occupancy,
                    transmembrane_buried=self.sampler.transmembrane_buried,
                    transmembrane_interface=self.sampler.transmembrane_interface,
                    global_transmembrane_label=self.sampler.global_transmembrane_label,
                    device=self.sampler.device,
                    subset_chains=self.sampler.subset_chains)

class PDB_dataset(torch.utils.data.Dataset):
    def __init__(self, IDs, loader, params):
        self.IDs = IDs
        self.loader = loader
        self.params = params

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        out = self.loader(ID, self.params)
        return out


def worker_init_fn(worker_id):
    np.random.seed()

def deduplicate_with_indices(lst):
    seen = set()
    deduped = []
    indices = []

    for i, item in enumerate(lst):
        # Convert to tuple to make it hashable
        item_tuple = tuple(item)
        if item_tuple not in seen:
            seen.add(item_tuple)
            deduped.append(item)
            indices.append(i)

    return indices, deduped


def loader_pdb(pdbid,params):

    PREFIX = "%s/pdb/%s/%s"%(params['METADIR'],pdbid[1:3],pdbid)
    pdb_path = "%s/pdb/%s/%s.pdb"%(params['PDBDIR'],pdbid[1:3],pdbid)
    
    # load metadata
    if not os.path.isfile(PREFIX+".pt") or not os.path.isfile(pdb_path):
        return {'seq': np.zeros(5)}
    meta = torch.load(PREFIX+".pt")
    asmb_ids, asmb_chains = deduplicate_with_indices(meta['asmb_chains'])
    chids = np.array(meta['chains'])

    # find candidate assemblies which contain chid chain
    asmb_candidates = set([a for a,b in zip(asmb_ids,asmb_chains)
                           ])

    # randomly pick one assembly from candidates
    idx = random.sample(list(asmb_candidates), 1)[0]

    # select chains which idx-th xform should be applied to
    s1 = set(meta['chains'])
    s2 = set(meta['asmb_chains'][idx].split(','))
    chains_k = s1&s2

    seq = ''
    if params['SUBSET']:
        for chain in chains_k:
            cid = meta['chains'].index(chain)
            seq += meta['seq'][cid][1].replace('-', '')
    else:
        for chain_seq in meta['seq']:
            if len(chain_seq[1].replace('-', '')) > 0:
                seq += chain_seq[1].replace('-', '')
            else:
                seq += chain_seq[0].replace('-', '')

    if params['SUBSET']:
        homo = set()
        for chidid in range(len(chids)):
            chid = chids[chidid]
            seqid = meta['tm'][chids==chid][0,:,1]
            homo = homo.union(set([ch_j for seqid_j,ch_j in zip(seqid[chidid+1:],chids[chidid+1:]) if seqid_j>0.7]))
        homo = list(homo)
        masked_list = [c for c in chains_k if c in homo]
    else:
        masked_list = []

    return {'filepath'    : pdb_path,
            'chains'    : list(chains_k),
            'masked_list': masked_list,
            'seq': seq,
            'name': pdbid
            }

def get_pdbs(data_loader, repeat=1, max_length=10000, num_units=1000000):
    pdb_dict_list = []
    for _ in range(repeat):
        for setp,t in enumerate(data_loader):
            if 'filepath' not in t.keys() or len(t['chains']) == 0 or len(t['seq'][0]) <= 32:
                # with open('/orcd/scratch/orcd/001/fosterb/pmpnn_experiments/ligandmpnn_potts_nomixed/log_pdbs.out', 'a') as f:
                #     f.write(str(t['filepath']) + ' ' + str(t['seq']) + ' ' + str(len(t['seq'])) + '\n')
                continue
            if len(t['seq'][0]) <= max_length and len(t['seq'][0]) > 0:
                pdb_dict_list.append(t)
            if len(pdb_dict_list) >= num_units:
                break
    return pdb_dict_list
