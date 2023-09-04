from pyteomics import proforma, mass
import pandas as pd
import numpy as np
import tqdm

max_length = 50
atomic = True
# Number of features
num_features = 13

# Define metadata about the sequence and amino acids
sequence_metadata = ["length", "in_sequence", "n_term_count", "c_term_count"]

# Map amino acids to their order
amino_acids_order = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}

feature_names = [
    "chemical_features",
    "diamino_chemical_features",
    "sequence_metadata",
    "one_hot",
    "is_modified"
]
feature_lengths = [
    num_features,
    num_features,
    len(sequence_metadata),
    len(amino_acids_order) + 1,  # +1 for is_modified row hot encoder
]

if atomic:
    atoms = ["C", "H", "O", "N", "P", "S"]

    feature_names = [
        "chemical_features",
        "diamino_chemical_features",
        "atoms",
        "sequence_metadata",
        "one_hot",
        "is_modified"
    ]

    feature_lengths = [
        num_features,
        num_features,
        len(atoms),
        len(sequence_metadata),
        len(amino_acids_order) + 1,  # +1 for is_modified row hot encoder
    ]


# Calculate feature indices based on lengths
feature_indices = {
    name: (sum(feature_lengths[: i]), sum(feature_lengths[: i + 1]))
    for i, name in enumerate(feature_names)
}

# Total number of channels
num_channels = sum(feature_lengths)

print(feature_indices)

def aa_atomic_composition_array():
    return {aa: np.array(
        [mass.std_aa_comp[aa][atom] for atom in atoms],
        dtype=np.float32)
        for aa in "ACDEFGHIKLMNPQRSTVWY"}

# Function to get chemical features for amino acids
def aa_chemical_feature(mask=None):

    df_aminoacids = pd.read_csv('aa_to_struct_to_features_20230804_4newMod.csv')

    if mask:
        df_aminoacids[df_aminoacids.columns[[mask]]] = 0
    # Convert the DataFrame to a dictionary with AAs as keys and chemical features as lists
    amino_acids = df_aminoacids.set_index('AA').T.to_dict('list')
    # Prepare a dictionary of feature arrays for each AA
    features_arrays = {
        aa: np.array(features, dtype=np.float32)
        for aa, features in amino_acids.items()
    }
    return features_arrays


# Function to get modification features
def mod_chemical_features(mask=None):
    # Return modifications with their features
    df = pd.read_csv('./struct_to_features_20230804_4newMod.csv')
    if mask:
        df[df.columns[mask]] = 0
    # Transpose the DataFrame and set the 'name' column as the index
    df = df.set_index('name').T
    # Convert the DataFrame to a dictionary of modifications with their chemical features
    modified = df.to_dict('list')
    dic = {}
    for key in modified:
        main_key, sub_key = key.split('#')
        # setdefault works as it check if key is in not in dict and allocate a dict to it
        dic.setdefault(main_key, {})[sub_key] = dict(zip(df.index, modified[key]))
    return dic


def peptide_parser(peptide):
    modifications = []
    parsed_sequence, modifiers = proforma.parse(peptide)
    sequence = "".join([aa for aa, _ in parsed_sequence])

    for loc, (_, mods) in enumerate(parsed_sequence):
        if mods:
            modifications.append(":".join([str(loc + 1), mods[0].name]))
    modifications = "|".join(modifications)

    return parsed_sequence, modifiers, sequence, modifications

# Function to create an empty array for encoding
def empty_array():
    return np.zeros(shape=(num_channels, max_length + 2), dtype=np.float32)


# Function to encode sequence and modification information
def encode_sequence_and_modification(sequence, parsed_sequence, modifications_dict, aa_to_feature, n_term=None,
                                     c_term=None):
    encoded = empty_array()
    start_index = feature_indices["chemical_features"][0]

    # Encoding the sequence
    for j, aa in enumerate(sequence):  # Encoding the sequence
        encoded[start_index: start_index + num_features, j + 1] = aa_to_feature[aa]
    # Encoding the modifications
    for loc, (aa, mods) in enumerate(parsed_sequence):  # Encoding the modification
        if mods:
            for mod in mods:
                name = mod.name
                encoded[start_index: start_index + num_features, loc + 1] = list(modifications_dict[name][aa].values())
        if n_term:
            for mod in n_term:
                name = mod.name
                encoded[start_index: start_index + num_features, 0] = list(modifications_dict[name][sequence[0]].values())
        if c_term:
            loc = len(parsed_sequence) + 1
            for mod in c_term:
                name = mod.name
                encoded[start_index: start_index + num_features, loc] = list(modifications_dict[name][sequence[-1]].values())

    return encoded

# Function to encode diamino sequence and modification information
def encode_diamino_sequence_and_modification(encode_seq_mod):
    encoded = empty_array()
    start_index = feature_indices["chemical_features"][0]
    start_index_diamino = feature_indices["diamino_chemical_features"][0]
    counter = 1

    for loc in range(1, len(encode_seq_mod.T) - 1, 2):
        encoded[start_index_diamino:start_index_diamino + num_features, counter] = encode_seq_mod[
                                                                                   start_index:start_index + num_features,
                                                                                   loc + 1] + encode_seq_mod[
                                                                                              start_index:start_index + num_features,
                                                                                              loc]
        counter += 1

    return encoded

def encode_sequence_and_modification_atomic(sequence, parsed_sequence, amino_acids_atoms, n_term=None, c_term=None):
    encoded = empty_array()
    start_index = feature_indices["atoms"][0]

    for j, aa in enumerate(sequence):  # Encoding the sequence
        encoded[start_index: start_index + len(atoms), j + 1] = amino_acids_atoms[aa]

    for loc, (aa, mods) in enumerate(parsed_sequence):  # Encoding the modification
        if mods:
            for mod in mods:
                mod_comp = mod.composition
                encoded[start_index: start_index + len(atoms), loc + 1] += [mod_comp[a] for a in atoms]
        if n_term:  # Encoding n_term
            for mod in n_term:
                mod_comp = mod.composition
                encoded[start_index:start_index + len(atoms), 0] += [mod_comp[a] for a in atoms]
        if c_term:  # Encoding c_term
            loc = len(parsed_sequence) + 1
            for mod in c_term:
                mod_comp = mod.composition
                encoded[start_index: start_index + len(atoms), loc] += [mod_comp[a] for a in atoms]

    return encoded

# Function to encode sequence metadata
def encode_sequence_metadata(sequence):
    encoded = empty_array()
    start_index = feature_indices["sequence_metadata"][0]
    seq_len = len(sequence)

    for i, m in enumerate(sequence_metadata):
        if m == "length":
            encoded[start_index + i, :] = seq_len / max_length
        elif m == "in_sequence":
            encoded[start_index + i, : seq_len + 2] = 1
        elif m == "n_term_count":
            encoded[start_index + i, : seq_len + 2] = (
                    np.arange(seq_len + 2) / seq_len
            )
        elif m == "c_term_count":
            encoded[start_index + i, : seq_len + 2] = (
                    np.arange(seq_len + 1, -1, -1) / seq_len
            )

    return encoded

# Function to encode sequence using one-hot encoding
def encode_sequence_one_hot(sequence):
    encoded = empty_array()
    start_index = feature_indices["one_hot"][0]

    for j, aa in enumerate(sequence):
        i = amino_acids_order[aa]
        encoded[start_index + i, j + 1] = 1
    return encoded

# Function to encode whether an amino acid is modified
def encode_mdf_is_modified(modifications):
    encoded = empty_array()
    start_index = feature_indices["one_hot"][1] - 1

    if modifications:
        mod_locs = [int(m.split(":")[0]) for m in modifications.split("|")]
        for j in mod_locs:
            encoded[start_index, j] = 1
    return encoded

# Function to convert a peptide to a matrix representation
def peptide_to_matrix(peptide, zero_column=None):
    parsed_sequence, modifiers, sequence, modifications = peptide_parser(peptide)
    modifications_dict = mod_chemical_features(mask=zero_column)
    aa_to_feature = aa_chemical_feature(mask=zero_column)
    amino_acids_atoms = aa_atomic_composition_array()
    encode_seq_mod = encode_sequence_and_modification(
        sequence, parsed_sequence, modifications_dict, aa_to_feature, modifiers["n_term"], modifiers["c_term"])
    encode_di_seq_mod = encode_diamino_sequence_and_modification(encode_seq_mod=encode_seq_mod)
    if atomic:
        encode_seq_mod_atomic = encode_sequence_and_modification_atomic(
            sequence, parsed_sequence, amino_acids_atoms, modifiers["n_term"], modifiers["c_term"])
    encode_seq_meta = encode_sequence_metadata(sequence)

    seq_hot = encode_sequence_one_hot(sequence)

    mod_hot = encode_mdf_is_modified(modifications)

    if not atomic:
        peptide_encoded = (
                encode_seq_mod
                + encode_di_seq_mod
                + encode_seq_meta
                + seq_hot
                + mod_hot
        )

    if atomic:
        peptide_encoded = (
                encode_seq_mod
                + encode_di_seq_mod
                + encode_seq_mod_atomic
                + encode_seq_meta
                + seq_hot
                + mod_hot
        )

    return peptide_encoded

# Function to convert a DataFrame to a matrix representation
def df_to_matrix(seqs, df, zero_column=None):
    seqs_encoded = []
    tr = []
    prediction = []
    errors = []
    modifications_dict = mod_chemical_features(mask=zero_column)
    aa_to_feature = aa_chemical_feature(mask=zero_column)
    amino_acids_atoms = aa_atomic_composition_array()
    for idx, peptide in tqdm.tqdm(enumerate(seqs)):
        try:
            parsed_sequence, modifiers, sequence, modifications = peptide_parser(peptide)
            encode_seq_mod = encode_sequence_and_modification(
                sequence, parsed_sequence, modifications_dict, aa_to_feature, modifiers["n_term"], modifiers["c_term"])
            encode_di_seq_mod = encode_diamino_sequence_and_modification(encode_seq_mod=encode_seq_mod)
            if atomic:
                encode_seq_mod_atomic = encode_sequence_and_modification_atomic(
                    sequence, parsed_sequence, amino_acids_atoms, modifiers["n_term"], modifiers["c_term"])
            encode_seq_meta = encode_sequence_metadata(sequence)

            seq_hot = encode_sequence_one_hot(sequence)

            mod_hot = encode_mdf_is_modified(modifications)
        except:
            errors.append([peptide, idx])
            continue
        if not atomic:
            peptide_encoded = (
                    encode_seq_mod
                    + encode_di_seq_mod
                    + encode_seq_meta
                    + seq_hot
                    + mod_hot
            )
        if atomic:
            peptide_encoded = (
                    encode_seq_mod
                    + encode_di_seq_mod
                    + encode_seq_mod_atomic
                    + encode_seq_meta
                    + seq_hot
                    + mod_hot
            )
        seqs_encoded.append(peptide_encoded)
        tr.append(df['tr'][idx])  # tr or tr_norm
        prediction.append(df['predictions'][idx])
    seqs_stack = np.stack(seqs_encoded)
    return seqs_stack, tr, prediction, errors
