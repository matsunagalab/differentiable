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
ff_output_xml_filepath = args[2]
ff_output_txt_filepath = args[3]
maximum_update_ratio = float(args[4]) #最大でffのパラメータをどの程度変えるか

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

#create forcefield
phi_ff_param_updated = dict()
psi_ff_param_updated = dict()

for key, value in phi_ff_param_original.items():
    #keyで条件分岐
    #type
    if re.search(r'type.*', key):
        phi_ff_param_updated[key] = value
        continue
    #periodicity
    if re.search(r'periodicity.*', key):
        phi_ff_param_updated[key] = value
        continue
    #phase
    if re.search(r'phase.*', key):
        phi_ff_param_updated[key] = value
        continue
    #k
    if re.search(r'k.*', key):
        k = float(value)
        #[-maximum_update_ratio, maximum_update_ratio]の範囲の乱数
        rnd = random.uniform(-maximum_update_ratio, maximum_update_ratio)
        k = k * (1 + rnd)
        phi_ff_param_updated[key] = str(k)
        continue

for key, value in psi_ff_param_original.items():
    #keyで条件分岐
    #type
    if re.search(r'type.*', key):
        psi_ff_param_updated[key] = value
        continue
    #periodicity
    if re.search(r'periodicity.*', key):
        psi_ff_param_updated[key] = value
        continue
    #phase
    if re.search(r'phase.*', key):
        psi_ff_param_updated[key] = value
        continue
    #k
    if re.search(r'k.*', key):
        k = float(value)
        #[-maximum_update_ratio, maximum_update_ratio]の範囲の乱数
        rnd = random.uniform(-maximum_update_ratio, maximum_update_ratio) 
        k = k * (1 + rnd)
        psi_ff_param_updated[key] = str(k)
        continue

#create xml file
tree = copy.deepcopy(ET.parse(ff_input_filepath)) #xml formatのff
root = tree.getroot()

# XMLから'PeriodicTorsionForce'を検索
for torsion in root.iter('PeriodicTorsionForce'):

    #各要素の中から、phiとpsiに関するパラメータを検索
    for ff_local in torsion:
        #print(list(ff_local.attrib.values())[0:4]) [0:4]でスライスすることで、アトムタイプを取り出す
        #phiに関係する二面角のパラメータを取り出す
        if list(ff_local.attrib.values())[0:4] == list(itertools.chain.from_iterable(phi_atom_type)):
            ff_local.attrib = phi_ff_param_updated
            
        #psiに関係する二面角のパラメータを取り出す
        if list(ff_local.attrib.values())[0:4] == list(itertools.chain.from_iterable(psi_atom_type)):        
            ff_local.attrib = psi_ff_param_updated

#output xml
tree.write(ff_output_xml_filepath)

#output txt
with open(ff_output_txt_filepath, 'w') as f:
    #phiに関係する情報
    f.write('PHI_PARAMETER\n')
    for key,value in phi_ff_param_updated.items():
    	f.write(f'{key}\t{value}\n')
    #psiに関係する情報
    f.write('PSI_PARAMETER\n')
    for key,value in psi_ff_param_updated.items():
    	f.write(f'{key}\t{value}\n')





