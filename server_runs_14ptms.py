
import os

mods = ['Acetyl','Carbamidomethyl','Crotonyl','Deamidated','Dimethyl','Formyl','Malonyl','Methyl','Nitro','Oxidation',
       'Phospho','Propionyl','Succinyl','Trimethyl']
# mods = ['Carbamidomethyl','Nitro','Oxidation','Phospho']

for m in mods:

    # os.system(f"python training_14ptms_4branch_diamino13_opt.py {m} ")
    os.system(f"python training_14ptms_diamino13.py {m} ")

