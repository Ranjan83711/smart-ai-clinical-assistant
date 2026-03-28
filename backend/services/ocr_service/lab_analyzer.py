from groq import Groq
import os
from dotenv import load_dotenv
import re

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def numbers(s):
    return [float(x) for x in re.findall(r'\d+\.?\d*', s)]

def find_range(nums):
    if len(nums) < 3:
        return None

    value = nums[0]
    for i in range(1,len(nums)):
        for j in range(i+1,len(nums)):
            low=min(nums[i],nums[j])
            high=max(nums[i],nums[j])
            if low < value < high:
                return value,low,high
    return None


def analyze_lab_report(text):

    lines=[l.strip() for l in text.split("\n") if l.strip()]
    findings=[]
    normal_tests=[]

    for i,line in enumerate(lines):

        if not re.search(r'[a-zA-Z]',line):
            continue

        context=line
        for k in range(1,4):
            if i+k<len(lines):
                context+=" "+lines[i+k]

        nums=numbers(context)
        result=find_range(nums)

        if result:
            value,low,high=result
            test=line[:25]

            if value<low:
                findings.append(f"{test} LOW ({value})")
            elif value>high:
                findings.append(f"{test} HIGH ({value})")
            else:
                normal_tests.append(test)

    # ---------- LLM EXPLANATION ----------
    structured_data = f"""
Normal Tests:
{", ".join(normal_tests)}

Abnormal Tests:
{", ".join(findings) if findings else "None"}
"""

    prompt = f"""
You are a medical AI assistant.

Your task is to extract and summarize ONLY meaningful medical values from a noisy lab report.

IGNORE:
- Hospital names
- Patient details
- IDs
- Random OCR errors
- Unclear or broken words

FOCUS ONLY ON:
- Blood test values (WBC, Hemoglobin, Platelets, etc.)
- Valid numbers with units

RULES:
- Do NOT include "Not clearly detected"
- Skip unclear values completely
- Keep output clean and minimal
- No long paragraphs
- No unnecessary labels

FORMAT:

🧾 Blood Test Summary  

✅ Overall Status  
(one short line summary)

🔬 Key Results  
- Parameter: Value → Normal/High/Low

💡 Interpretation  
- 2–3 short insights

🩺 Recommendation  
- 1–2 simple suggestions

LAB REPORT:
{structured_data}
"""

    res = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"user","content":prompt}],
        temperature=0.2
    )

    return res.choices[0].message.content