"""
Built-in PulseIQ health assistant.

Answers from a small, hand-written knowledge base, so the chat works on any
server with no AI model, API key or internet access. It is used whenever no
language model (Groq or Ollama) is available, and as a fallback if one fails.

Matching: the message is lower-cased and split into words; small typos are
corrected against the knowledge-base keywords (e.g. "bo" -> "bp"); each topic is
scored by how many of its keywords appear, and the best topic's answer is used.
Emergency symptoms are always checked first.
"""
import re
import difflib

DISCLAIMER = "This is general information, not a diagnosis. Please check with a doctor for advice about you."

# Phrases that must always get the emergency answer, whatever else is asked.
EMERGENCY = [
    "chest pain", "chest tightness", "pain in my chest", "cant breathe", "can't breathe",
    "cannot breathe", "short of breath", "shortness of breath", "difficulty breathing",
    "face drooping", "slurred speech", "arm weakness", "numb on one side", "stroke",
    "fainted", "passed out", "unconscious", "heart attack", "severe headache",
    "vision loss", "confused suddenly", "180/120", "180 / 120",
]
EMERGENCY_REPLY = (
    "⚠ These can be signs of an emergency. Please call emergency services (112 in India) "
    "or go to the nearest emergency department now; don't wait for an app reading. "
    "Warning signs include chest pain, severe breathlessness, sudden weakness or numbness on one side, "
    "drooping face, slurred speech, fainting, or a reading of 180/120 mmHg or higher with symptoms. "
    "You can use the Hospital Finder page to locate the nearest hospital."
)

# (keywords, answer). Keywords are single words or short phrases.
TOPICS = [
    (["bp", "blood pressure", "pressure"],
     "Blood pressure (BP) is the force of blood pushing against your artery walls as the heart pumps. "
     "It is written as two numbers in mmHg, for example 120/80: the first is systolic (SBP, the peak "
     "pressure when the heart beats) and the second is diastolic (DBP, the lowest pressure between beats). "
     "PulseIQ also shows MAP, the average pressure over the whole heartbeat."),
    (["systolic", "sbp", "top number", "upper number", "first number"],
     "Systolic pressure (SBP) is the top number of a BP reading: the highest pressure in your arteries "
     "when the heart contracts. For most adults, under 120 mmHg is considered normal, and 140 or above "
     "is in the high range."),
    (["diastolic", "dbp", "bottom number", "lower number", "second number"],
     "Diastolic pressure (DBP) is the bottom number of a BP reading: the pressure in your arteries while "
     "the heart relaxes between beats. For most adults, under 80 mmHg is considered normal, and 90 or "
     "above is in the high range."),
    (["map", "mean arterial", "mean pressure", "average pressure"],
     "MAP (mean arterial pressure) is the average pressure in your arteries over one whole heartbeat. "
     "It reflects how well blood reaches your organs; roughly 70–100 mmHg is typical for adults. "
     "A common estimate is (SBP + 2 × DBP) ÷ 3, but PulseIQ's model predicts MAP directly."),
    (["mmhg", "millimetres of mercury", "millimeters of mercury", "unit"],
     "mmHg means millimetres of mercury, the standard unit for blood pressure. A reading of 120 mmHg is "
     "the pressure that could push a column of mercury 120 mm high (1 mmHg ≈ 133 pascals)."),
    (["normal", "range", "ranges", "category", "categories", "band", "bands", "chart", "healthy", "ideal", "good bp", "good blood pressure"],
     "PulseIQ uses these adult bands (graded on systolic; a high diastolic can move a reading up a band): "
     "Low: under 90 · Normal: 90–119 · Borderline: 120–129 · Elevated: 130–139 (or DBP 80–89) · "
     "High: 140 or above (or DBP 90+). One reading isn't enough to label anyone; doctors look at "
     "repeated measurements."),
    (["high", "hypertension", "elevated", "raised", "high bp", "high blood pressure"],
     "High blood pressure (hypertension) means the pressure in your arteries stays raised, usually "
     "140/90 mmHg or above on repeated readings. It often has no symptoms but strains the heart, brain, "
     "kidneys and eyes over time. Common contributors are high salt intake, excess weight, inactivity, "
     "alcohol, smoking, stress, age and family history. A doctor confirms it with several readings."),
    (["low", "hypotension", "dizzy", "dizziness", "lightheaded", "light headed"],
     "Low blood pressure (hypotension) is usually a systolic reading under 90 mmHg. Many healthy people "
     "run low without problems; it matters when it causes dizziness, blurred vision, fainting or "
     "tiredness. Getting up slowly, drinking enough fluids and talking to a doctor (especially if you "
     "take BP medicines) can help."),
    (["lower", "reduce", "decrease", "control", "manage", "bring down", "prevent", "tips", "improve"],
     "Proven ways to lower blood pressure: cut salt to under 5 g a day (about one teaspoon), eat more "
     "fruit, vegetables, pulses and whole grains (the DASH pattern), be active at least 150 minutes a "
     "week, keep a healthy weight, limit alcohol, stop smoking, and sleep 7–9 hours. If a doctor has "
     "prescribed medicine, keep taking it as directed."),
    (["salt", "sodium", "diet", "food", "foods", "eat", "eating", "dash", "potassium"],
     "For blood pressure, the biggest diet lever is salt: aim for under 5 g a day and watch pickles, "
     "papad, namkeen, instant noodles, sauces and restaurant food. Potassium-rich foods such as bananas, "
     "leafy greens, dal, curd and coconut water help balance sodium (unless a doctor has told you to "
     "limit potassium)."),
    (["exercise", "walk", "walking", "workout", "yoga", "physical activity", "gym", "running"],
     "Regular activity lowers blood pressure: aim for 150 minutes a week of moderate exercise, such as "
     "30 minutes of brisk walking on five days, plus some strength work twice a week. Yoga and breathing "
     "exercises can also help. If your BP is very high or you have heart problems, check with a doctor "
     "before starting intense exercise."),
    (["stress", "anxiety", "tension", "sleep", "relax", "meditation"],
     "Stress raises blood pressure for short periods, and poor sleep is linked to higher BP over time. "
     "Slow breathing, meditation, regular exercise and 7–9 hours of sleep can help. If stress or "
     "anxiety feels overwhelming, talking to a doctor or counsellor is a good step."),
    (["caffeine", "coffee", "tea", "alcohol", "drink", "drinking", "smoking", "cigarette", "tobacco"],
     "Caffeine can raise BP briefly, so avoid it for 30 minutes before a measurement. Alcohol raises BP "
     "over time; keep it low or avoid it. Smoking and tobacco raise BP with every use and damage the "
     "arteries, so quitting is one of the best things for your heart."),
    (["measure", "measuring", "check", "checking", "cuff", "monitor", "correctly", "accurate reading"],
     "To measure BP accurately with a cuff: sit quietly for 5 minutes, back supported and feet flat; "
     "rest your arm at heart level; use the right cuff size on bare skin; don't talk; avoid caffeine, "
     "exercise and smoking for 30 minutes before. Take two readings a minute apart and note both."),
    (["heart rate", "pulse", "bpm", "hr", "heartbeat"],
     "A normal resting heart rate for adults is about 60–100 beats per minute (fit people can be lower). "
     "PulseIQ calculates heart rate by counting the pulse peaks in your PPG signal."),
    (["ppg", "photoplethysmogram", "photoplethysmography", "pulse oximeter", "finger sensor", "sensor"],
     "PPG (photoplethysmogram) is the pulse wave measured by a light sensor, like the clip on your "
     "fingertip. With each heartbeat the finger fills with a little more blood, so the reflected light "
     "rises and falls. PulseIQ estimates blood pressure from the shape of this wave and its first and "
     "second derivatives (VPG and APG)."),
    (["vpg", "apg", "derivative", "velocity", "acceleration"],
     "VPG and APG are calculated from the PPG wave: VPG is its first derivative (how fast the wave "
     "changes) and APG is the second derivative (how fast that speed changes). They make small details "
     "of the pulse shape easier for the model to see."),
    (["cuffless", "how does pulseiq", "how does it work", "how it works", "how do you", "predict",
      "prediction", "estimate", "model", "ai", "resnet", "lstm", "bilstm", "deep learning"],
     "PulseIQ estimates SBP, DBP and MAP from a 12-second PPG recording, without a cuff. A deep-learning "
     "model (a 1D ResNet followed by a bidirectional LSTM) reads the pulse wave and its derivatives and "
     "outputs the three values in mmHg. It was trained on ICU recordings where the true pressure came "
     "from an arterial line."),
    (["accurate", "accuracy", "reliable", "trust", "error", "how good", "correct", "wrong"],
     "PulseIQ is a research demonstrator, not a medical device. On patients it had never seen, its "
     "systolic estimate was off by about 18 mmHg on average, and it does not meet the clinical AAMI/BHS "
     "accuracy standards. Use it to explore the technology, and confirm any reading with a validated cuff."),
    (["explanation", "explainability", "integrated gradients", "ig", "shap",
      "attention", "xai", "reason", "why this reading"],
     "The 'Why this reading?' page explains each estimate two ways. Integrated Gradients shows which part "
     "of the heartbeat (rise, peak & dip, or slow fall) the model relied on most. SHAP shows which "
     "measurable pulse features, such as heart rate or pulse width, pushed the value up or down, in mmHg."),
    (["medicine", "medicines", "medication", "medications", "tablet", "tablets", "pill", "pills",
      "dose", "amlodipine", "telmisartan", "losartan", "drug", "drugs"],
     "I can't advise on starting, stopping or changing any medicine; please follow your doctor's "
     "prescription and ask them or a pharmacist about side effects. Never stop BP medicine suddenly on "
     "your own. The Medicine Reminders page can help you take doses on time."),
    (["headache", "nosebleed", "symptom", "symptoms", "blurred", "vision"],
     "High blood pressure usually has no symptoms. Headaches, nosebleeds or blurred vision can have many "
     "causes; if they are severe or sudden, or come with a very high reading (180/120 or more), seek "
     "medical care urgently. Otherwise, mention them to your doctor."),
    (["dashboard", "page", "pages", "navigate", "where", "feature", "features", "app", "use pulseiq",
      "history", "vault", "reminder", "reminders", "icu", "hospital", "finder", "messages", "profile"],
     "In PulseIQ: the Dashboard shows your latest estimated BP and heart rate; ICU Monitor shows the live "
     "PPG, VPG and APG waves; 'Why this reading?' explains the estimate; History lists past readings; "
     "Medical Vault stores your documents; Medicine Reminders tracks doses; Hospital Finder locates "
     "nearby hospitals; and Messages lets you chat with your doctor."),
    (["doctor", "see a doctor", "consult", "appointment", "when to see"],
     "See a doctor if your readings are repeatedly 140/90 mmHg or higher, if you have symptoms such as "
     "dizziness, fainting, severe headaches or vision changes, or before changing any medicine. "
     "Get emergency care for chest pain, severe breathlessness, stroke signs, or a reading of 180/120 or "
     "more with symptoms. You can message your doctor from the Messages page."),
    (["hello", "hi", "hey", "namaste", "good morning", "good evening", "who are you", "what can you do"],
     "Hi! I'm the PulseIQ Health Assistant. I can explain blood pressure (SBP, DBP, MAP), healthy ranges, "
     "lifestyle tips, how to measure BP correctly, and how PulseIQ and its explanations work. "
     "What would you like to know?"),
    (["thanks", "thank you", "thankyou", "ok", "okay", "great", "cool"],
     "You're welcome! Ask me anything else about blood pressure or PulseIQ."),
]

MY_READING = ["my reading", "my bp", "my blood pressure", "my result", "my results", "my value",
              "my pressure", "my numbers", "my latest", "is my", "am i"]

STOP = {"what", "is", "are", "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "my", "me",
        "i", "you", "it", "this", "that", "how", "do", "does", "can", "could", "should", "would", "may",
        "might", "will", "be", "was", "were", "with", "about", "tell", "explain", "please", "pls", "plz",
        "mean", "means", "meaning", "define", "kya", "hai", "bad", "good", "best", "worst", "get", "has", "have", "had", "any", "all", "too", "very", "much", "more", "most", "than", "then", "when", "your", "its", "if", "at", "by", "so", "why"}

_VOCAB = sorted({w for kws, _ in TOPICS for kw in kws for w in kw.split()})
_SHORT = [w for w in _VOCAB if len(w) <= 4 and w not in STOP]


def _one_edit(a, b):
    """True if a and b differ by exactly one substitution, insertion or deletion."""
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    if len(a) > len(b):
        a, b = b, a
    return any(b[:i] + b[i + 1:] == a for i in range(len(b)))


def _normalise(text):
    words = re.findall(r"[a-z0-9/]+", text.lower())
    fixed = []
    for w in words:
        if w in _VOCAB or w in STOP or w.isdigit():
            fixed.append(w)
            continue
        if len(w) <= 4:
            near = [v for v in _SHORT if _one_edit(w, v)]
            fixed.append(near[0] if len(near) == 1 else w)
        else:
            near = difflib.get_close_matches(w, [v for v in _VOCAB if v not in STOP], n=1, cutoff=0.8)
            fixed.append(near[0] if near else w)
    return " ".join(fixed)


def _best_topic(text):
    padded = f" {text} "
    best, best_score = None, 0.0
    for idx, (kws, _) in enumerate(TOPICS):
        score = 0.0
        for kw in kws:
            if f" {kw} " in padded:
                score += 1.5 if " " in kw else 1.0   # phrases count a little more
        if idx == 0:
            score *= 0.3          # the general "what is BP" topic loses ties to specific ones
        if score > best_score:
            best, best_score = idx, score
    return best


def answer(message, history=None, reading=None):
    """Return a reply for `message`.

    history: list of {"role","content"} (used only to follow up on the last topic).
    reading: optional dict {"SBP","DBP","MAP","band"} for the signed-in patient.
    """
    raw = (message or "").lower()
    if any(p in raw for p in EMERGENCY):
        return EMERGENCY_REPLY

    text = _normalise(raw)

    if reading and any(p in raw for p in MY_READING):
        return (f"Your most recent PulseIQ estimate is about {reading['SBP']}/{reading['DBP']} mmHg "
                f"(MAP {reading['MAP']}), which falls in the '{reading['band']}' band. "
                "It is an estimate from your pulse wave, not a cuff measurement, so confirm it with a "
                "validated cuff before acting on it. " + DISCLAIMER)

    idx = _best_topic(text)
    follow_up = re.search(r"\b(more|elaborate|continue|detail|details|go on|explain again)\b", raw)
    if idx is None and history and follow_up:
        # "tell me more" style follow-ups: reuse the last user question's topic
        for m in reversed(history):
            if m.get("role") == "user":
                idx = _best_topic(_normalise(m.get("content", "").lower()))
                if idx is not None:
                    break
    if idx is None:
        return ("I'm a built-in assistant with a fixed set of topics, and I didn't catch that one. "
                "I can help with: what BP, SBP, DBP and MAP mean; normal ranges; high or low BP; "
                "lowering BP through diet, exercise and sleep; measuring BP correctly; heart rate; "
                "PPG; how PulseIQ works and how accurate it is; and the 'Why this reading?' page. "
                "Try asking, for example, 'What is a normal BP?'")
    reply = TOPICS[idx][1]
    kws = TOPICS[idx][0]
    if any(k in kws for k in ("high", "low", "lower", "medicine", "headache", "normal")):
        reply += " " + DISCLAIMER
    return reply
