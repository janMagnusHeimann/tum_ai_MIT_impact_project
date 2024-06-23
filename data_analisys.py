from classes import *
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import re
from collections import defaultdict
import periodictable

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(119, 360)
        self.fc2 = nn.Linear(360, 180)
        self.fc3 = nn.Linear(180, 100)
        self.fc4 = nn.Linear(100, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x

def train(model, train_loader, criterion, optimizer, num_epochs=1000):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, label in train_loader:
            optimizer.zero_grad()
            output = model(inputs)

            loss = criterion(output, label)

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

def test(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, label in test_loader:
            output = model(inputs)
            loss = criterion(output, label)
            total_loss += loss.item() * inputs.size(0)
    avg_loss = total_loss / len(test_loader.dataset)
    print(f'Evaluation Loss: {avg_loss:.4f}')

def PreprocessElementCombinationData(data: List[FormulaPart]):
    return sorted(data, key= lambda item: item.material)


def GetElementInfoForIndex(data: List[FormulaPart], index: int, property : str = "Element"):
    if(index >= len(data)): return None
    if(property.lower()[0]=='e'):
        return data[index].material
    else:
        return data[index].amount

def flatten(data: ReactionEntry):
    # Pre-processing
    rxn_lhs = PreprocessElementCombinationData(data.reaction.left_side)
    rxn_rhs = PreprocessElementCombinationData(data.reaction.right_side)
    payload = {}
    for i in range(len(rxn_lhs)):
      payload[f'LHS {i}'] = [GetElementInfoForIndex(rxn_lhs, i, "Element"), GetElementInfoForIndex(rxn_lhs, i, "Amount")]

    for i in range(len(rxn_rhs)):
      payload[f'RHS {i}'] = [GetElementInfoForIndex(rxn_rhs, i, "Element"), GetElementInfoForIndex(rxn_lhs, i, "Amount")]

    return payload

# periodic_table = periodictable

# for element in periodic_table.composition:
#     print(f'{element.symbol}: {element.name} - Atomic number: {element.number}')

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
    'Ds': 110, 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118,
    'dummy':119
}

def parse_formula(formula):

    element_pattern = re.compile(r'([A-Z][a-z]*)(\d*\.?\d*)')
    group_pattern = re.compile(r'\((.*?)\)(\d*\.?\d*)')
    composition = defaultdict(float)

    def add_composition(formula_part, multiplier=1):
        for match in element_pattern.finditer(formula_part):
            element = match.group(1)
            # print("ahahdfa" , element)
            # if element == ".":
            #   continue
            quantity = match.group(2)

            if quantity == ".":
              quantity = 1
            else:
              quantity = float(quantity) if quantity else 1.0
            composition[element] += quantity * multiplier

    while True:
        match = group_pattern.search(formula)
        if not match:
            break
        group_formula = match.group(1)
        group_multiplier = match.group(2)
        group_multiplier = float(group_multiplier) if group_multiplier else 1.0

        add_composition(group_formula, group_multiplier)

        formula = formula[:match.start()] + formula[match.end():]

    add_composition(formula)
    return composition

def encode_compositional_vector(formula):
    composition = parse_formula(formula)
    # print("composition", composition)
    total_atoms = sum(composition.values())
    vector = [0] * len(periodic_table)  # Create a zero vector with length equal to the number of composition

    for element, count in composition.items():
        try:
          index = periodic_table[element]
        except:
          index = periodic_table['dummy']
        vector[index - 1] = count / total_atoms  # Convert to fraction

    return vector

# Example usage
# formulas = ["H2O", "Ni(OH)2", "C6H12O6", "Ca3(PO4)2"]
# for formula in formulas:
#     vector = encode_compositional_vector(formula)
#     print(f"Compositional vector for {formula}: {vector}")
#     print(len(vector))

############# LOAD DATA ################

filename = r'data/sol-gel_dataset_20200713.json'
filename = r'data/solid-state_dataset_20200713.json'
filename = r'data/solid-state_dataset_2019-12-03.json'

filedata = open(filename, mode='r').read()
jsonParse = json.loads(filedata)

reactions = [from_dict(reaction, ReactionEntry) for reaction in jsonParse['reactions']]
print(flatten(reactions[0]))
############# FILTER SINTERED AND CALCINATION TEMPS ################

parsed = []
# print(reactions[3].operations[1])

calc_max_values = []
sint_max_values = []

input = []
zero = torch.zeros(1)
for j in range(len(reactions)):#
  flattened = flatten(reactions[j])
  formulas = [str(flattened[key][0]) for key in flattened.keys()]

  vectors = []
  for formula in formulas:
    # print("formula", formula)
    vector = encode_compositional_vector(formula)
    vectors.append(torch.tensor(vector))
    # print(f"Compositional vector for {formula}: {vector}")
  vectors = torch.stack(vectors, dim=0).sum(dim=0)
  vectors = torch.tensor(vectors)

  # print(reactions[j].operations)
  calc_max_values = []
  sint_max_values = []
  for i in range(len(reactions[j].operations)):
    operations = reactions[j].operations[i]
    # print("operations.token", operations.token)
    if operations.token == "calcined":
      # print("calcined")
      if operations.conditions is not None and operations.conditions.heating_temperature is not None:
        calc_max_value = operations.conditions.heating_temperature[0].max_value
        if calc_max_value is not None:
          calc_max_values.append(torch.tensor(calc_max_value))
        else:
          calc_max_values.append(zero)

    elif operations.token == "sintered":
      # print("sintered")
      if operations.conditions is not None and operations.conditions.heating_temperature is not None:
        sint_max_value = operations.conditions.heating_temperature[0].max_value
        if sint_max_value is not None:
          sint_max_values.append(torch.tensor(sint_max_value))
        else:
          sint_max_values.append(zero)

    else:
      # print("smth else")
      calc_max_values.append(zero)
      sint_max_values.append(zero)



  calc_max_values = torch.tensor(calc_max_values)
  calc_max_mean = torch.mean(calc_max_values)
  # print("calc_max_mean", calc_max_mean)

  sint_max_values = torch.tensor(sint_max_values)
  sint_max_mean = torch.mean(sint_max_values)
  # print("sint_max_mean", sint_max_mean)

  input.append([vector, torch.tensor(calc_max_mean), torch.tensor(sint_max_mean)])

# print(len(input))

TASK = "sintered"

dset = []
X = []
y = []
if TASK == "calcined":
  for i in range(len(input)):
    if input[i][1] != 0 and torch.isnan(input[i][1]) == False:
      dset.append([input[i][0],input[i][1]])
  for i in range(len(dset)):
    X.append(dset[i][0])
    y.append(dset[i][1])
elif TASK == "sintered":
  for i in range(len(input)):
    if input[i][2] != 0 and torch.isnan(input[i][2]) == False:
      dset.append([input[i][0],input[i][2]])
  for i in range(len(dset)):
    X.append(dset[i][0])
    y.append(dset[i][1])
X = torch.tensor(X)
y = torch.tensor(y)
print(X.shape)
print(y.shape)


model = NN()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

batch_size = 1
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

train(model, train_loader, criterion, optimizer)
test(model, test_loader, criterion)


