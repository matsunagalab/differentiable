import mdtraj as md
import numpy as np
import pandas as pd
import csv
import pprint
import matplotlib.pyplot as plt
import os
import itertools
import sys

#コマンドライン引数
args = sys.argv

pdb_filepath = args[1]
dcd_filepath = args[2]
output_filepath = args[3]

t= md.load(pdb_filepath)
top = t.topology

traj = md.load_dcd(dcd_filepath, top=top) #mdtrajでdcdファイルを読み込む
phi_indices = list(itertools.chain.from_iterable(md.compute_phi(traj)[0])) #phiのindex
psi_indices = list(itertools.chain.from_iterable(md.compute_psi(traj)[0])) #psiのindex
angles = md.compute_dihedrals(traj, [phi_indices, psi_indices]) #dihedralのtrajデータ

#output txt file
with open(output_filepath, 'w') as f:
    for i in range(angles.shape[0]):
        s = f'{angles[i,0]}\t{angles[i,1]}\n'
        f.write(s)