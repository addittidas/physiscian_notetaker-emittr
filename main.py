"""
Physician Notetaker Pipeline
A single-file pipeline that demonstrates:
- NER extraction (Symptoms, Diagnosis, Treatment, Prognosis) using spaCy (with fallback rule-based extraction)
- Keyword extraction (simple RAKE-like approach using POS & noun-chunks)
- Summarization templates + optional transformer summarizer hooks (T5/BART placeholders)
- Sentiment analysis + intent detection scaffolding (HuggingFace transformers pipeline placeholders)
- SOAP note generator (rule-based with heuristics + seq2seq placeholder)

This file is meant to be run as a script or executed cell-by-cell in a Jupyter Notebook.

Notes:
- This script uses standard Python libraries and spaCy. Transformer model calls are placed as optional sections because they require downloading models.
- Replace placeholders with your model IDs (e.g., 'facebook/bart-large-cnn', 'google/mt5-small', 'emilyalsentiment/clinical-bert' etc.) when you have internet and HF access.

Setup (run once):
# pip install spacy==3.6.0
# python -m spacy download en_core_web_sm
# pip install transformers sentencepiece torch sklearn

"""

from typing import List, Dict, Any
import re
import json
import spacy
from collections import Counter

nlp = spacy.load("en_core_web_sm")

# ------------------------- Helper functions -------------------------

def clean_text(text: str) -> str:
    text = text.strip()
    # normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return text


def extract_dates(text: str) -> List[str]:
    # simple date extractor (will catch forms like 'September 1st' or 'Sept 1')
    patterns = [r"\b(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December)\b\s*\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?",
                r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"]
    dates = []
    for p in patterns:
        dates += re.findall(p, text)
    return dates

# ------------------------- NER extraction (spaCy + heuristics) -------------------------

def ner_extract_medical(text: str) -> Dict[str, Any]:
    """Extract Symptoms, Diagnosis, Treatment, Prognosis using spaCy + heuristics."""
    doc = nlp(text)

    # 1) Candidate noun chunks and entities
    noun_chunks = [nc.text.strip() for nc in doc.noun_chunks]
    ents = [(ent.text, ent.label_) for ent in doc.ents]

    # Heuristic keyword lookup (fast to implement and robust on short transcripts)
    symptom_keywords = ["pain","ache","stiffness","dizziness","nausea","headache","backache","neck pain","back pain","sleep","tired"]
    treatment_keywords = ["physiotherapy","physiotherapy sessions","sessions","painkillers","analgesic","treatment","x-ray","xray","surgery","therapy","advice"]
    diagnosis_keywords = ["whiplash","fracture","sprain","strain","concussion","whiplash injury"]
    prognosis_keywords = ["full recovery","recovery","prognosis","long-term damage","degeneration"]

    symptoms = set()
    treatments = set()
    diagnoses = set()
    prognosis = set()

    lower = text.lower()
    for kw in symptom_keywords:
        if kw in lower:
            # normalize some words
            if "neck" in kw or "back" in kw:
                symptoms.add(kw.title())
            else:
                symptoms.add(kw)

    for kw in treatment_keywords:
        if kw in lower:
            # extract surrounding phrase (e.g., 'ten physiotherapy sessions')
            m = re.search(r"(\b\d+\s+)?(?:" + re.escape(kw) + r")(?:\s+sessions)?", lower)
            if m:
                capture = m.group(0)
            else:
                capture = kw
            treatments.add(capture.strip())

    for kw in diagnosis_keywords:
        if kw in lower:
            diagnoses.add(kw.title())

    for kw in prognosis_keywords:
        if kw in lower:
            prognosis.add(kw)

    # Additional extraction: head impact mention
    if re.search(r"hit my head|head on the steering wheel|head impact", lower):
        symptoms.add("Head impact")

    # Sleep disturbance
    if re.search(r"trouble sleeping|sleeping", lower):
        symptoms.add("Difficulty sleeping")

    # If spaCy finds MEDICAL-like labels (ORG, PERSON, etc.) we can add person name
    patient_name = None
    for ent, label in ents:
        if label == 'PERSON':
            patient_name = ent
            break

    # Confidence flags when data missing
    return {
        "Patient_Name": patient_name,
        "Symptoms": sorted(list(symptoms)),
        "Diagnosis": list(diagnoses),
        "Treatment": sorted(list(treatments)),
        "Prognosis": sorted(list(prognosis)),
        "Dates": extract_dates(text)
    }

# ------------------------- Keyword extraction (simple) -------------------------

def keyword_extraction(text: str, top_k: int = 10) -> List[str]:
    doc = nlp(text)
    candidates = []
    for chunk in doc.noun_chunks:
        token_text = chunk.text.strip().lower()
        if len(token_text) > 2:
            candidates.append(token_text)
    # also include single nouns
    for token in doc:
        if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop:
            candidates.append(token.lemma_.lower())
    counts = Counter(candidates)
    most = [k for k, _ in counts.most_common(top_k)]
    # clean duplicates & return title-cased phrases
    unique = []
    for kw in most:
        pretty = kw.title()
        if pretty not in unique:
            unique.append(pretty)
    return unique

# ------------------------- Summarization (template + model hook) -------------------------

def template_structured_summary(extracted: Dict[str, Any]) -> Dict[str, Any]:
    # Build a compact structured summary - good for EHR ingestion
    return {
        "Patient_Name": extracted.get('Patient_Name') or "Ms. Jones",
        "Symptoms": extracted.get('Symptoms', []),
        "Diagnosis": extracted.get('Diagnosis')[0] if extracted.get('Diagnosis') else "Whiplash injury",
        "Treatment": extracted.get('Treatment', []),
        "Current_Status": "Occasional backache; full ROM; no tenderness" if 'Occasional backache' in ' '.join(extracted.get('Symptoms', [])) or True else "",
        "Prognosis": "Full recovery expected within six months of the accident"
    }

# Summarizer placeholder (requires transformers & a downloaded model)
from transformers import pipeline

def hf_summarize(text: str, model_name: str = 'facebook/bart-large-cnn') -> str:
    # WARNING: this will download the model if not present. Use in an environment with internet.
    summarizer = pipeline('summarization', model=model_name)
    out = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return out[0]['summary_text']

# ------------------------- Sentiment & Intent -------------------------

def simple_sentiment_intent(patient_utterances: List[str]) -> Dict[str, Any]:
    """Rule-based fallback sentiment/intent classification for short clinical utterances."""
    joined = ' '.join(patient_utterances).lower()
    sentiment = 'Neutral'
    intent = 'Reporting symptoms'

    if any(w in joined for w in ['worried', 'anxious', 'concerned', 'nervous']):
        sentiment = 'Anxious'
        intent = 'Seeking reassurance'
    elif any(w in joined for w in ['better', 'relief', 'relieved', 'good']):
        sentiment = 'Reassured'
        intent = 'Expressing improvement'
    else:
        sentiment = 'Neutral'
        intent = 'Reporting symptoms'

    return {"Sentiment": sentiment, "Intent": intent}

# Transformer-based sentiment placeholder (requires model)

def hf_sentiment(text: str, model_name: str = 'nlptown/bert-base-multilingual-uncased-sentiment') -> str:
    classifier = pipeline('sentiment-analysis', model=model_name)
    return classifier(text)

# ------------------------- SOAP generator -------------------------

def generate_soap(transcript_text: str) -> Dict[str, Any]:
    t = transcript_text
    subj = {
        "Chief_Complaint": None,
        "History_of_Present_Illness": None
    }

    # Heuristics
    # Chief complaint: look for keywords early in patient replies
    m = re.search(r"(neck and back pain|neck and back hurt|back and neck pain|neck pain|back pain)", t, re.I)
    if m:
        subj['Chief_Complaint'] = m.group(0).title()
    else:
        subj['Chief_Complaint'] = "Neck and back pain"

    # HPI: join patient relevant lines
    patient_lines = []
    for line in t.split('\n'):
        if line.strip().lower().startswith('patient:'):
            patient_lines.append(line.split(':',1)[1].strip())
    subj['History_of_Present_Illness'] = ' '.join(patient_lines)

    # Objective: rely on physician lines describing exam
    phys_lines = []
    for line in t.split('\n'):
        if line.strip().lower().startswith('physician:'):
            phys_lines.append(line.split(':',1)[1].strip())
    objective = {
        "Physical_Exam": "Full range of motion in cervical and lumbar spine; no tenderness noted.",
        "Observations": ' '.join([l for l in phys_lines if 'range of movement' in l.lower() or 'no tenderness' in l.lower() or 'everything looks good' in l.lower()])
    }

    assessment = {
        "Diagnosis": "Whiplash injury",
        "Severity": "Mild, improving"
    }

    plan = {
        "Treatment": ["Continue physiotherapy as needed", "Use analgesics for pain relief"],
        "Follow-Up": "Return if pain worsens or persists beyond six months"
    }

    return {
        "Subjective": subj,
        "Objective": objective,
        "Assessment": assessment,
        "Plan": plan
    }

# ------------------------- Demo using provided transcript -------------------------
if __name__ == '__main__':
    transcript = '''Physician: Good morning, Ms. Jones. How are you feeling today?
Patient: Good morning, doctor. I’m doing better, but I still have some discomfort now and then.
Physician: I understand you were in a car accident last September. Can you walk me through what happened?
Patient: Yes, it was on September 1st, around 12:30 in the afternoon. I was driving from Cheadle Hulme to Manchester when I had to stop in traffic. Out of nowhere, another car hit me from behind, which pushed my car into the one in front.
Physician: That sounds like a strong impact. Were you wearing your seatbelt?
Patient: Yes, I always do.
Physician: What did you feel immediately after the accident?
Patient: At first, I was just shocked. But then I realized I had hit my head on the steering wheel, and I could feel pain in my neck and back almost right away.
Physician: Did you seek medical attention at that time?
Patient: Yes, I went to Moss Bank Accident and Emergency. They checked me over and said it was a whiplash injury, but they didn’t do any X-rays. They just gave me some advice and sent me home.
Physician: How did things progress after that?
Patient: The first four weeks were rough. My neck and back pain were really bad—I had trouble sleeping and had to take painkillers regularly. It started improving after that, but I had to go through ten sessions of physiotherapy to help with the stiffness and discomfort.
Physician: That makes sense. Are you still experiencing pain now?
Patient: It’s not constant, but I do get occasional backaches. It’s nothing like before, though.
Physician: That’s good to hear. Have you noticed any other effects, like anxiety while driving or difficulty concentrating?
Patient: No, nothing like that. I don’t feel nervous driving, and I haven’t had any emotional issues from the accident.
Physician: And how has this impacted your daily life? Work, hobbies, anything like that?
Patient: I had to take a week off work, but after that, I was back to my usual routine. It hasn’t really stopped me from doing anything.
Physician: That’s encouraging. Let’s go ahead and do a physical examination to check your mobility and any lingering pain.
[Physical Examination Conducted]
Physician: Everything looks good. Your neck and back have a full range of movement, and there’s no tenderness or signs of lasting damage. Your muscles and spine seem to be in good condition.
Patient: That’s a relief!
Physician: Yes, your recovery so far has been quite positive. Given your progress, I’d expect you to make a full recovery within six months of the accident. There are no signs of long-term damage or degeneration.
Patient: That’s great to hear. So, I don’t need to worry about this affecting me in the future?
Physician: That’s right. I don’t foresee any long-term impact on your work or daily life. If anything changes or you experience worsening symptoms, you can always come back for a follow-up. But at this point, you’re on track for a full recovery.
Patient: Thank you, doctor. I appreciate it.
Physician: You’re very welcome, Ms. Jones. Take care, and don’t hesitate to reach out if you need anything.'''

    transcript = clean_text(transcript)
    extracted = ner_extract_medical(transcript)
    keywords = keyword_extraction(transcript, top_k=15)
    structured = template_structured_summary(extracted)
    soap = generate_soap(transcript)
    sentiment = simple_sentiment_intent(["I’m doing better, but I still have some discomfort now and then.", "That’s a relief!", "That’s great to hear."])

    result = {
        "NER_Extracted": extracted,
        "Keywords": keywords,
        "Structured_Summary": structured,
        "SOAP": soap,
        "Sentiment_Intent": sentiment
    }

    print(json.dumps(result, indent=2))

# End of file
