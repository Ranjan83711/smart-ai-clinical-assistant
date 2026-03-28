import re

def extract_medicines(lines):

    meds=[]

    for line in lines:

        dose = re.findall(r'(\d-\d-\d|od|bd|hs|tid|qid)', line.lower())
        strength = re.findall(r'\d+\s?mg', line.lower())

        meds.append({
            "line": line,
            "dosage": dose[0] if dose else None,
            "strength": strength[0] if strength else None
        })

    return meds