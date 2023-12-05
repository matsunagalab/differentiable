import xml.etree.ElementTree as ET
import itertools
import random
import copy
import re
import os
import sys

#コマンドライン引数
args = sys.argv

#path
ff_input_filepath  = args[1]
ff_output_txt_filepath = args[2]

#atom_type
phi_atom_type = [['C'], ['N'], ['CT'], ['C']]
psi_atom_type = [['N'], ['CT'], ['C'], ['N']]

#input xml format ff
tree = copy.deepcopy(ET.parse(ff_input_filepath)) #xml formatのff
root = tree.getroot()

# XMLから'PeriodicTorsionForce'を検索
for torsion in root.iter('PeriodicTorsionForce'):

    #各要素の中から、phiとpsiに関するパラメータを検索
    for ff_local in torsion:
        #print(list(ff_local.attrib.values())[0:4]) [0:4]でスライスすることで、アトムタイプを取り出す
        #phiに関係する二面角のパラメータを取り出す
        if list(ff_local.attrib.values())[0:4] == list(itertools.chain.from_iterable(phi_atom_type)):
            phi_ff_element = ff_local #xml.etree.ElementTree.Element型でphiに関連する行を抜き出す
            
        #psiに関係する二面角のパラメータを取り出す
        if list(ff_local.attrib.values())[0:4] == list(itertools.chain.from_iterable(psi_atom_type)):        
            psi_ff_element = ff_local

#辞書型でパラメータの値を取り出す
phi_ff_param_original = copy.deepcopy(phi_ff_element.attrib)
psi_ff_param_original = copy.deepcopy(psi_ff_element.attrib)

with open(ff_output_txt_filepath, 'w') as f:
    #phiに関係する情報
    f.write('PHI_PARAMETER\n')
    for key,value in phi_ff_param_original.items():
    	f.write(f'{key}\t{value}\n')
    #psiに関係する情報
    f.write('PSI_PARAMETER\n')
    for key,value in psi_ff_param_original.items():
    	f.write(f'{key}\t{value}\n')