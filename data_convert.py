import pandas as pd
import numpy as np
import tqdm
import utilities_diamino13_new_meta as utilities
from pathlib import Path

main_file = './Data_files/Reporting Summary.csv'
dataset_name = 'ProteomeTools'


all_datasets = {'ATLANTIS SILICA': 'atlantis', 'SCX': "scx", 'Yeast 2h': "yeast2h", 'HeLa HF': "helahf",
                'ProteomeTools PTM': "proteometoolsptm", 'Arabidopsis': "arabidopsis", 'LUNA SILICA': "lunasilica",
                'Xbridge': "xbridge", 'Pancreas': "pancreas", 'Plasma lumos 2h': "plasma2h",
                'HeLa DeepRT': "heladeeprt",
                'SWATH library': "swath", 'HeLa Lumos 1h': "hela1h", 'Yeast DeepRT': "yeastdeeprt",
                'ProteomeTools': "proteometools", 'HeLa Lumos 2h': "hela2h", 'DIA HF': "diahf",
                'LUNA HILIC': "lunahilic",
                'Plasma lumos 1h': "plasma1h", 'Yeast 1h': "yeast1h"}

dataset_save_name = all_datasets[dataset_name]

"""PAY ATTENTION TO UTILITIES IMPORT, DATASET NAME, PATH SAVE,& DATATYPE AT THE END"""

# 'ATLANTIS SILICA', 'SCX', 'Yeast 2h', 'HeLa HF', 'ProteomeTools PTM', 'Arabidopsis', 'LUNA SILICA', 'Xbridge',
# 'Pancreas', 'Plasma lumos 2h', 'HeLa DeepRT', 'SWATH library', 'HeLa Lumos 1h', 'Yeast DeepRT', 'ProteomeTools',
# 'HeLa Lumos 2h', 'DIA HF', 'LUNA HILIC', 'Plasma lumos 1h', 'Yeast 1h'


# Changing the sequence to desired form
# Pay attention to the index, +1 can be needed depending if it counts AAs from 0 or 1
def reform_seq(seq, mod):
    mod_list = [m for m in mod.split('|')]
    mod_list_tuple = []
    if mod == '':
        return (seq)
    else:
        while mod_list:
            mod_list_tuple.append((int(mod_list.pop(0)), mod_list.pop(0)))
        mod_list_tuple.sort()
        while mod_list_tuple != []:
            tuple_mod = mod_list_tuple.pop()
            modification = tuple_mod[1]
            index = tuple_mod[0]
            seq = seq[:int(index)] + '[' + modification + ']' + seq[int(index):]
        return (seq)


# extract a dataset from whole data based on types
def data_separator(file, dataset, type):
    index = []
    sequence = []
    mod = []
    retention = []
    prediction = []
    df = pd.read_csv(file, keep_default_na=False)

    for idx, seq in enumerate(df['seq']):
        if df['data set'][idx] == dataset and df['set_type'][idx] == type:
            index.append(idx)
            sequence.append(seq)
            mod.append(df['modifications'][idx])
            retention.append(df['tr'][idx])
            prediction.append((df['predictions'][idx]))

    data_frame = pd.DataFrame()
    data_frame['index'] = index
    data_frame['seq'] = sequence
    data_frame['modifications'] = mod
    data_frame['tr'] = retention
    data_frame['predictions'] = prediction

    return data_frame


# Transforming each Peptide to a Matrix for training

def seq_to_matrix(data, df, column_mask=None):
    results = []
    tr = []
    prediction = []
    errors = []
    for idx, seq in tqdm.tqdm(enumerate(data)):
        try:
            if column_mask:
                result = utilities.peptide_to_matrix(seq, column_mask)
            else:
                result = utilities.peptide_to_matrix(seq)
            results.append(result)
            tr.append(df['tr'][idx])  # tr or tr_norm
            prediction.append(df['predictions'][idx])
        except KeyError as e:
            errors.append([seq, idx])
        except Exception as e:
            errors.append([seq, idx, e])
    results_stack = np.stack(results)
    print(len(tr), len(data), 'Usable data=', len(tr) / len(data) * 100, '%')
    return results_stack, tr, prediction, errors


def dataframe_to_matrix(data, df, column_mask=None):

    results, tr, prediction, errors = utilities.df_to_matrix(data, df, column_mask)

    print(len(tr), len(data), 'Usable data=', len(tr) / len(data) * 100, '%')
    return results, tr, prediction, errors


def main(file=main_file, dataset=dataset_name, dataset_save=dataset_save_name, column_mask=None):
    df_train = data_separator(file, dataset, 'train')
    df_test = data_separator(file, dataset, 'test')
    df_val = data_separator(file, dataset, 'validation')

    # Normalize the data based on min and max

    maxx, minn = df_train['tr'].max(), df_train['tr'].min()
    for i in [df_test, df_val]:
        if i['tr'].min() < minn:
            minn = int(i['tr'].min())
        if i['tr'].max() > maxx:
            maxx = int(i['tr'].max())
    print(f'dataset: {dataset}', 'min:', minn, 'max:', maxx)

    for i in [df_train, df_test, df_val]:
        i['tr_norm'] = (i['tr'] - minn) / (maxx - minn)
    my_train = [reform_seq(df_train['seq'][i], m) for i, m in enumerate(df_train['modifications'])]
    my_test = [reform_seq(df_test['seq'][i], m) for i, m in enumerate(df_test['modifications'])]
    my_eval = [reform_seq(df_val['seq'][i], m) for i, m in enumerate(df_val['modifications'])]

    # Create X and y for train, test and validation sets
    #
    # train_x, train_y, train_prediction, train_error = seq_to_matrix(my_train, df_train, column_mask)
    # test_x, test_y, test_prediction, test_error = seq_to_matrix(my_test, df_test, column_mask)
    # val_x, val_y, val_prediction, val_error = seq_to_matrix(my_eval, df_val, column_mask)


    train_x, train_y, train_prediction, train_error = dataframe_to_matrix(my_train, df_train, column_mask)
    test_x, test_y, test_prediction, test_error = dataframe_to_matrix(my_test, df_test, column_mask)
    val_x, val_y, val_prediction, val_error = dataframe_to_matrix(my_eval, df_val, column_mask)

    print('Shape of train set: ', train_x.shape)
    print('Shape of test set: ', test_x.shape)
    print('Shape of val set: ', val_x.shape)

    path_save = './data_matrix/ProteomeTools'#/' + dataset_save + '/'
    data_type = '' #'diamino' + '/' + str(column_mask)
    # save the matrix
    Path(path_save + data_type).mkdir(parents=True, exist_ok=True)

    np.save(path_save + data_type + '/train_x', train_x)
    np.save(path_save + data_type + '/train_y', train_y)
    np.save(path_save + data_type + '/test_x', test_x)
    np.save(path_save + data_type + '/test_y', test_y)
    np.save(path_save + data_type + '/val_x', val_x)
    np.save(path_save + data_type + '/val_y', val_y)



if __name__ == '__main__':

    main()
    # for k, v in all_datasets.items():
    #     main(dataset=k, dataset_save=v)

    # for number_columns in range(1,15):
    #     main(column_mask=number_columns)

