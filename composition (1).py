import re
from collections import defaultdict
import numpy as np

# Complete periodic table with element positions (index starts from 1)
periodic_table = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
    'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
    'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
    'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
    'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
    'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
    'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
    'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
    'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109,
    'Ds': 110, 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118
}

# Function to parse the chemical formula
def parse_formula(formula):
    # Pattern to match elements, counts, and groups with parentheses
    pattern = r'([A-Z][a-z]?|\([^\(\)]+\))(\d*)'
    matches = re.findall(pattern, formula)
    composition = defaultdict(int)
    
    def add_composition(comp_dict, multiplier=1):
        for element, count in matches:
            if element.startswith('('):
                # Remove parentheses and parse the inner formula
                inner_formula = element[1:-1]
                inner_composition = parse_formula(inner_formula)
                multiplier=int(count)
                for inner_element, inner_count in inner_composition.items():
                    comp_dict[inner_element] += inner_count * multiplier
            else:
                if count == '':
                    count = 1
                else:
                    count = int(count)
                comp_dict[element] += count * multiplier
    
    add_composition(composition)
    
    return composition

def parse_prettyformula(formula):
    pattern = r'([A-Z][a-z]?)(\d*)'
    matches = re.findall(pattern, formula)
    composition=defaultdict(int)
    for element, count in matches:
        if count=='' or count == None:
            count=1
        composition[element]= int(count)
    return composition


# Function to create the compositional vector
def encode_compositional_vector(formula):
    composition = parse_prettyformula(formula)
    total_atoms = sum(composition.values())
    vector = [0] * len(periodic_table)  # Create a zero vector with length equal to the number of elements
    
    for element, count in composition.items():
        index = periodic_table[element]
        vector[index - 1] = count / total_atoms  # Convert to fraction
    
    return vector


# input: list of prettyformula type materials
# output: list of compositional vectors
def convertFormulaToVector(formula:list):
    output=list()
    for material in formula:
        output.append(encode_compositional_vector(material))
    return output

# concatenates precursors and targets to a 1 dimensional array
#input: list of compositional vectors
def concatenate(precursors:list, targets:list):
    maxPrecursors= 6
    maxTargets=6
    vectorLength=118

    precursors.append((maxPrecursors-len(precursors))*(vectorLength*[0]))
    targets.append((maxTargets-len(targets))*(vectorLength*[0]))

    return np.concatenate([np.concatenate(precursors),np.concatenate(targets)])

#input: list of compositional vectors
def sumCompositional(precursors:list, targets:list):
    prec=list()
    for i in range(len(precursors[0])):
        num=0
        for k in range(len(precursors)):
            num+=precursors[k][i]
        prec.append(num)

    targ=list()
    for i in range(len(targets[0])):
        num=0
        for k in range(len(targets)):
            num+=targets[k][i]
        targ.append(num)
            
    return np.concatenate([prec,targ])






    

# Example usage
formulas = ["H2O", "NiO2H2", "C6H12O6", "Ca3P2O8"]
prec=["H2O", "NaCl"]
targ=["Fe3Na2","Ga4Ge2"]

precursors=convertFormulaToVector(prec)
targets=convertFormulaToVector(targ)
conc=concatenate(precursors,targets)
suM=sumCompositional(precursors,targets)

print(len(suM))

