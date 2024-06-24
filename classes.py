from typing import List, Dict, Optional
import json

def from_dict(data, cls):
    """
    Convert a dictionary to a class instance.
    """
    if isinstance(data, list):
        return [from_dict(item, cls) for item in data]
    elif isinstance(data, dict):
        if cls == Operation:
            conditions = None
            if 'conditions' in data:
                conditions = from_dict(data.pop('conditions'), OperationConditions)
            return cls(**data, conditions=conditions)
        elif cls == OperationConditions:
            heating_temperature = from_dict(data.get('heating_temperature'), OperationValue) if data.get('heating_temperature') else None
            heating_time = from_dict(data.get('heating_time'), OperationValue) if data.get('heating_time') else None
            return cls(
                heating_temperature=heating_temperature,
                heating_time=heating_time,
                heating_atmosphere=data.get('heating_atmosphere'),
                mixing_device=data.get('mixing_device'),
                mixing_media=data.get('mixing_media')
            )
        elif cls == Formula:
            left_side = from_dict(data['left_side'], FormulaPart)
            right_side = from_dict(data['right_side'], FormulaPart)
            return cls(left_side=left_side, right_side=right_side, element_substitution=data['element_substitution'])
        elif cls == Material:
            composition = None
            if 'composition' in data:
                composition = from_dict(data['composition'], Composition)
            
            try: 
                phase=data.get('phase')
            except:
                phase=None
            
            try:
                material_name=data["material_name"]
            except:
                material_name=None
            
            try:
                is_acronym=data["is_acronym"]
            except: 
                is_acronym=None
            return cls(
                material_string=data['material_string'],
                material_formula=data['material_formula'],
                material_name=material_name,
                phase=phase,
                is_acronym=is_acronym,
                composition=composition,
                amount_vars=data.get('amount_vars', {}),
                element_vars=data.get('element_vars', {}),
                additives=data.get('additives', []),
                oxygen_deficiency=data.get('oxygen_deficiency')
            )
        elif cls == ReactionEntry:
            reaction = from_dict(data['reaction'], Formula)
            target = from_dict(data['target'], Material)
            precursors = from_dict(data['precursors'], Material)
            operations = from_dict(data['operations'], Operation)
            try:
                synsthesis_type = data["synthesis_type"]
            except: 
                synsthesis_type=None
            return cls(
                doi=data['doi'],
                paragraph_string=data['paragraph_string'],
                synthesis_type=synsthesis_type,
                reaction_string=data['reaction_string'],
                reaction=reaction,
                targets_string=data['targets_string'],
                target=target,
                precursors=precursors,
                operations=operations
            )
        else:
            return cls(**data)
    else:
        return data

class FormulaPart:
    def __init__(self, amount: str, material: str):
        self.amount = amount
        self.material = material

    def __repr__(self):
        return f"FormulaPart(amount={self.amount!r}, material={self.material!r})"

    def __str__(self):
        return json.dumps(self.__dict__, indent=2)


class Formula:
    def __init__(self, left_side: List[FormulaPart], right_side: List[FormulaPart], element_substitution: Dict[str, str]):
        self.left_side = left_side
        self.right_side = right_side
        self.element_substitution = element_substitution

    def __repr__(self):
        return (f"Formula(left_side={self.left_side!r}, right_side={self.right_side!r}, "
                f"element_substitution={self.element_substitution!r})")

    def __str__(self):
        return json.dumps(self.__dict__, indent=2, default=str)


class Composition:
    def __init__(self, formula: str, amount: str, elements: Dict[str, str]):
        self.formula = formula
        self.amount = amount
        self.elements = elements

    def __repr__(self):
        return (f"Composition(formula={self.formula!r}, amount={self.amount!r}, "
                f"elements={self.elements!r})")

    def __str__(self):
        return json.dumps(self.__dict__, indent=2)


class Material:
    def __init__(self, material_string: str, material_formula: str, material_name: str, phase: Optional[str], is_acronym: bool, composition: List[Composition], amount_vars: Dict[str, List[str]], element_vars: Dict[str, List[str]], additives: List[str], oxygen_deficiency: Optional[str]):
        self.material_string = material_string
        self.material_formula = material_formula
        self.material_name = material_name
        self.phase = phase
        self.is_acronym = is_acronym
        self.composition = composition
        self.amount_vars = amount_vars
        self.element_vars = element_vars
        self.additives = additives
        self.oxygen_deficiency = oxygen_deficiency

    def __repr__(self):
        return (f"Material(material_string={self.material_string!r}, material_formula={self.material_formula!r}, "
                f"material_name={self.material_name!r}, phase={self.phase!r}, is_acronym={self.is_acronym!r}, "
                f"composition={self.composition!r}, amount_vars={self.amount_vars!r}, "
                f"element_vars={self.element_vars!r}, additives={self.additives!r}, "
                f"oxygen_deficiency={self.oxygen_deficiency!r})")

    def __str__(self):
        return json.dumps(self.__dict__, indent=2, default=str)


class OperationValue:
    def __init__(self, min_value: float, max_value: float, values: List[float], units: str):
        self.min_value = min_value
        self.max_value = max_value
        self.values = values
        self.units = units

    def __repr__(self):
        return (f"OperationValue(min_value={self.min_value!r}, max_value={self.max_value!r}, "
                f"values={self.values!r}, units={self.units!r})")

    def __str__(self):
        return json.dumps(self.__dict__, indent=2)


class OperationConditions:
    def __init__(self, heating_temperature: Optional[List[OperationValue]], heating_time: Optional[List[OperationValue]], heating_atmosphere: Optional[str], mixing_device: Optional[str], mixing_media: Optional[str]):
        self.heating_temperature = heating_temperature
        self.heating_time = heating_time
        self.heating_atmosphere = heating_atmosphere
        self.mixing_device = mixing_device
        self.mixing_media = mixing_media

    def __repr__(self):
        return (f"OperationConditions(heating_temperature={self.heating_temperature!r}, "
                f"heating_time={self.heating_time!r}, heating_atmosphere={self.heating_atmosphere!r}, "
                f"mixing_device={self.mixing_device!r}, mixing_media={self.mixing_media!r})")

    def __str__(self):
        return json.dumps(self.__dict__, indent=2, default=str)


class Operation:
    def __init__(self, type: str, token: str, conditions: OperationConditions):
        self.type = type
        self.token = token
        self.conditions = conditions

    def __repr__(self):
        return (f"Operation(type={self.type!r}, token={self.token!r}, "
                f"conditions={self.conditions!r})")

    def __str__(self):
        return json.dumps(self.__dict__, indent=2, default=str)


class ReactionEntry:
    def __init__(self, doi: str, paragraph_string: str, synthesis_type: str, reaction_string: str, reaction: Formula, targets_string: List[str], target: Material, precursors: List[Material], operations: List[Operation]):
        self.doi = doi
        self.paragraph_string = paragraph_string
        self.synthesis_type = synthesis_type
        self.reaction_string = reaction_string
        self.reaction = reaction
        self.targets_string = targets_string
        self.target = target
        self.precursors = precursors
        self.operations = operations

    def __repr__(self):
        return (f"ReactionEntry(doi={self.doi!r}, paragraph_string={self.paragraph_string!r}, "
                f"synthesis_type={self.synthesis_type!r}, reaction_string={self.reaction_string!r}, "
                f"reaction={self.reaction!r}, targets_string={self.targets_string!r}, "
                f"target={self.target!r}, precursors={self.precursors!r}, "
                f"operations={self.operations!r})")

    def __str__(self):
        return json.dumps(self.__dict__, indent=2, default=str)


class Payload:
    def __init__(self, release_date: str, reactions: List[ReactionEntry]):
        self.release_date = release_date
        self.reactions = reactions

    def __repr__(self):
        return f"Payload(release_date={self.release_date!r}, reactions={self.reactions!r})"

    def __str__(self):
        return json.dumps(self.__dict__, indent=2, default=str)
