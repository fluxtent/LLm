"""Lightweight medical ontology helpers for MedBrief AI."""

from __future__ import annotations

import re
from dataclasses import dataclass


ICD10_CATEGORIES: dict[str, list[str]] = {
    "mental_and_behavioral": [
        "major depressive disorder",
        "generalized anxiety disorder",
        "panic disorder",
        "ptsd",
        "ocd",
        "bipolar disorder",
        "adhd",
        "substance use disorder",
    ],
    "cardiovascular": [
        "hypertension",
        "heart failure",
        "atrial fibrillation",
        "myocardial infarction",
        "stroke",
        "deep vein thrombosis",
    ],
    "endocrine": [
        "diabetes mellitus",
        "hypothyroidism",
        "hyperthyroidism",
        "cushing syndrome",
    ],
    "gastrointestinal": [
        "gerd",
        "ulcerative colitis",
        "crohn disease",
        "hepatomegaly",
    ],
}

DSM5_TERMS = {
    "depression",
    "anxiety",
    "panic attack",
    "mania",
    "psychosis",
    "dissociation",
    "depersonalization",
    "derealization",
    "obsession",
    "compulsion",
}

DRUG_VOCAB = {
    "sertraline",
    "fluoxetine",
    "citalopram",
    "escitalopram",
    "venlafaxine",
    "bupropion",
    "lamotrigine",
    "alprazolam",
    "clonazepam",
    "lorazepam",
    "diazepam",
    "metformin",
    "semaglutide",
    "empagliflozin",
    "warfarin",
    "apixaban",
    "dabigatran",
    "ibuprofen",
    "acetaminophen",
    "diphenhydramine",
    "cetirizine",
    "loratadine",
    "hydroxyzine",
}

SYMPTOM_VOCAB = {
    "chest pain",
    "shortness of breath",
    "weakness",
    "numbness",
    "headache",
    "fatigue",
    "fever",
    "dizziness",
    "nausea",
    "constipation",
    "weight gain",
    "cold intolerance",
    "brain fog",
}

PROCEDURE_VOCAB = {
    "mri",
    "ct scan",
    "x-ray",
    "colonoscopy",
    "endoscopy",
    "esophagogastroduodenoscopy",
    "biopsy",
    "cbc",
    "ekg",
}

EMERGENCY_TERMS = {
    "heart attack",
    "stroke",
    "can't breathe",
    "cannot breathe",
    "severe chest pain",
    "sudden weakness",
    "face drooping",
    "slurred speech",
    "blue lips",
}


@dataclass(frozen=True)
class MedicalContext:
    symptoms: list[str]
    drugs: list[str]
    diagnoses: list[str]
    procedures: list[str]
    emergency_terms: list[str]

    @property
    def has_medical_signal(self) -> bool:
        return any([self.symptoms, self.drugs, self.diagnoses, self.procedures, self.emergency_terms])


def _find_terms(text: str, vocabulary: set[str] | list[str]) -> list[str]:
    lowered = text.lower()
    found = [term for term in vocabulary if term in lowered]
    return sorted(set(found))


def detect_medical_context(text: str) -> MedicalContext:
    diagnoses = []
    for items in ICD10_CATEGORIES.values():
        diagnoses.extend(term for term in items if term in text.lower())
    diagnoses.extend(term for term in DSM5_TERMS if term in text.lower())

    return MedicalContext(
        symptoms=_find_terms(text, SYMPTOM_VOCAB),
        drugs=_find_terms(text, DRUG_VOCAB),
        diagnoses=sorted(set(diagnoses)),
        procedures=_find_terms(text, PROCEDURE_VOCAB),
        emergency_terms=_find_terms(text, EMERGENCY_TERMS),
    )


def emergency_medical_request(text: str) -> bool:
    return bool(detect_medical_context(text).emergency_terms)


def scrub_unapproved_drugs(response_text: str) -> tuple[str, bool]:
    tokens = set(re.findall(r"[a-zA-Z][a-zA-Z-]{2,}", response_text.lower()))
    mentioned_drugs = {token for token in tokens if token in DRUG_VOCAB}
    unknown_drug_like_terms = {token for token in tokens if token.endswith(("mab", "pril", "sartan", "oxetine")) and token not in DRUG_VOCAB}
    if not unknown_drug_like_terms:
        return response_text, False
    return (
        response_text
        + " I want to be careful not to overstate medication-specific information here, so a clinician or pharmacist should verify any drug-specific interpretation.",
        True,
    )


def terminology_instruction(terminology_preference: str) -> str:
    if terminology_preference == "professional":
        return "Use medically literate terminology when appropriate, while still being clear."
    return "Prefer plain-language medical explanations and define terms briefly when they appear."
