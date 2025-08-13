import pandas as pd
import random, textwrap, pathlib

random.seed(7)

topics = [
    ("Hypertension", [
        "First-line therapy often includes lifestyle changes such as reduced sodium intake and regular aerobic exercise.",
        "Thiazide diuretics, ACE inhibitors, ARBs, and calcium channel blockers are common medications.",
        "Blood pressure targets may vary by guideline and comorbidities."
    ]),
    ("Type 2 Diabetes", [
        "Metformin is commonly used as initial pharmacotherapy unless contraindicated.",
        "Lifestyle interventions (dietary pattern, weight loss, physical activity) improve glycemic control.",
        "Monitor for microvascular complications: retinopathy, nephropathy, neuropathy."
    ]),
    ("Asthma", [
        "Short-acting beta-agonists are for relief; inhaled corticosteroids are controller therapy.",
        "Trigger avoidance and inhaler technique education are key.",
        "Assess control with symptom frequency, night awakenings, and SABA use."
    ]),
    ("Hyperlipidemia", [
        "Statins reduce LDL and ASCVD risk; intensity depends on risk profile.",
        "Lifestyle modification complements pharmacotherapy.",
        "Consider non-statin agents if LDL targets unmet or statin intolerance."
    ]),
    ("Vaccination", [
        "Immunization schedules differ by age, risk factors, and pregnancy status.",
        "Shared decision-making for certain adult vaccines.",
        "Cold chain integrity is essential for vaccine efficacy."
    ]),
]

rows = []
for topic, bullets in topics:
    for i, b in enumerate(bullets, 1):
        passage = textwrap.fill(f"{topic}: {b}", 90)
        rows.append({
            "id": f"{topic.lower().replace(' ', '-')}-{i}",
            "topic": topic,
            "text": passage,
            "source": f"{topic.replace(' ', '_')}.md",
            "section": f"{topic} #{i}"
        })

pathlib.Path("db").mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv("db/medical_kb.csv", index=False)
print("Wrote db/medical_kb.csv with", len(rows), "rows")
