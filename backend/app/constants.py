"""Static product constants for the MedBrief production gateway."""

SUPPORTED_MODES = ("psych", "health", "crisis", "portfolio", "general")

SYSTEM_PROMPT = """
You are MedBrief AI, a calm, capable assistant for emotional support, health education, and general help.
Be direct, useful, and human. Do not diagnose, prescribe, or give medication dosing.
""".strip()

MODE_INSTRUCTIONS = {
    "psych": "Assume emotional pain unless injury is explicit. Validate first, stay grounded, keep it brief, avoid generic boilerplate, and ask one gentle follow-up question.",
    "health": "Answer the health question directly in plain language. Give general education only, not diagnosis or treatment.",
    "crisis": "Prioritize safety, empathy, and immediate crisis resources. Do not continue casual conversation until safety is addressed.",
    "portfolio": "Explain MedBrief AI clearly and credibly without hype.",
    "general": "Answer directly, clearly, and without filler.",
}

FRONTEND_FEATURE_FLAGS = {
    "apiKeysEnabled": False,
    "moodCheckEnabled": True,
    "memoryInsightsEnabled": True,
    "feedbackEnabled": True,
    "profileEnabled": True,
    "modeTooltipsEnabled": True,
}

CRISIS_RESOURCE_BLOCK = (
    "If you're in crisis right now, please call or text 988 in the US, "
    "contact local emergency services if you are in immediate danger, "
    "or reach out to an in-person crisis professional right away."
)

CRISIS_RESPONSE = (
    "I'm really glad you said something. Your safety matters more than anything else right now. "
    "Please call or text 988 right now if you're in the US, or contact your local emergency services "
    "if you might act on these thoughts. If you can, reach out to someone nearby and tell them you need "
    "support staying safe. You do not have to handle this alone."
)

MEDICATION_DOSING_RESPONSE = (
    "I can't help with medication dosing or tell you how much to take. "
    "That needs to come from a licensed clinician, pharmacist, or poison control professional who can "
    "consider your specific medication, health history, and safety risks. "
    "If you tell me the medication and what you're worried about, I can help with general information "
    "or suggest the safest next question to ask a professional."
)

DEGRADED_MODE_RESPONSE = (
    "MedBrief AI is taking a brief break. Please try again in a moment."
)

LOW_QUALITY_FALLBACK = (
    "I want to make sure I give you a thoughtful response. Could you tell me a bit more about what you mean?"
)

PSYCH_KEYWORDS = (
    "anxious",
    "depressed",
    "feeling",
    "overwhelmed",
    "struggling",
    "panic",
    "therapist",
    "burnout",
    "lonely",
    "stress",
    "hurt",
    "hurting",
    "grief",
    "lost",
)

HEALTH_KEYWORDS = (
    "symptom",
    "medication",
    "diagnosis",
    "doctor",
    "pain",
    "condition",
    "treatment",
    "hospital",
    "medical",
    "ssri",
)

PORTFOLIO_KEYWORDS = (
    "medbrief",
    "portfolio",
    "project",
    "website",
    "who made",
    "what is this",
    "about you",
    "product",
)

CRISIS_KEYWORDS = (
    "suicide",
    "suicidal",
    "kill myself",
    "end my life",
    "want to die",
    "hurt myself",
    "self harm",
    "self-harm",
    "better off dead",
    "no reason to live",
    "nothing to live for",
    "can't go on",
    "cannot go on",
    "don't know how to go on",
    "dont know how to go on",
    "i've lost so much",
    "ive lost so much",
    "im hurting",
    "i'm hurting",
    "no point in living",
    "no point anymore",
    "what's the point",
    "what is the point",
    "tired of everything",
    "nothing matters",
    "done fighting",
    "i give up",
    "can't do this anymore",
    "cant do this anymore",
    "nothing left",
    "lost the will",
    "so exhausted of living",
    "lost everything",
    "ive lost everything",
    "lost my will",
    "everyone would be better off",
    "disappear",
    "fade away",
    "never wake up",
)

HIGH_CONFIDENCE_CRISIS_KEYWORDS = (
    "suicide",
    "suicidal",
    "kill myself",
    "end my life",
    "want to die",
    "hurt myself",
    "self harm",
    "self-harm",
    "better off dead",
    "no reason to live",
    "nothing to live for",
    "can't go on",
    "cannot go on",
    "don't know how to go on",
    "dont know how to go on",
    "no point in living",
)

CRISIS_DISTRESS_SIGNALS = (
    "help",
    "alone",
    "hurting",
    "hurt",
    "hopeless",
    "lost",
    "can't",
    "cant",
    "done",
    "exhausted",
    "crying",
    "panic",
    "overwhelmed",
    "scared",
    "afraid",
)

MEDICAL_EMERGENCY_KEYWORDS = (
    "heart attack",
    "stroke",
    "slurred speech",
    "face drooping",
    "cannot breathe",
    "can't breathe",
    "severe chest pain",
)

MEDICATION_KEYWORDS = (
    "sertraline",
    "prozac",
    "fluoxetine",
    "lexapro",
    "escitalopram",
    "zoloft",
    "ibuprofen",
    "acetaminophen",
    "tylenol",
    "adderall",
    "xanax",
    "klonopin",
    "lamotrigine",
    "wellbutrin",
    "citalopram",
    "venlafaxine",
    "bupropion",
)

PROFANITY_TERMS = (
    "fuck",
    "fucking",
    "shit",
    "bitch",
    "asshole",
)

PRIVACY_DISCLAIMER = (
    "MedBrief AI is for educational purposes only. Do not share personally identifying health information. "
    "For diagnosis, treatment, or medication advice, consult a licensed clinician."
)
