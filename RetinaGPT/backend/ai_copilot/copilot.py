"""
ai_copilot/copilot.py — AI Copilot for Retinal Scans
======================================================
Natural language interface: doctors ask questions about a scan,
get intelligent answers grounded in the AI analysis results.

Two layers:
    1. CLIP semantic similarity  — query ↔ retinal findings
    2. Structured reasoning      — map findings to clinical answers

Questions the Copilot handles:
    "What abnormalities do you see?"
    "Is this worse than last visit?"
    "Should I refer this patient?"
    "What is the confidence level?"
    "Explain why you graded this as Moderate DR."
    "Are there any signs of AMD?"
    "What lesions are present?"
    "How urgent is this case?"
"""

from __future__ import annotations
import re
from typing import Dict, Any, Optional, List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Intent Classifier
# ─────────────────────────────────────────────────────────────────────────────

INTENT_PATTERNS = {
    "dr_grade": [
        r"dr grade", r"diabetic retinopathy", r"retinopathy grade",
        r"how severe", r"how bad", r"what grade", r"grade.*dr",
    ],
    "refer": [
        r"refer", r"urgent", r"send to specialist", r"ophthalmolog",
        r"should i refer", r"need referral", r"needs to be seen",
    ],
    "lesions": [
        r"lesion", r"microaneurysm", r"hemorrhage", r"exudate",
        r"cotton.wool", r"neovascular", r"drusen", r"what.*see",
        r"abnormali", r"findings",
    ],
    "explain": [
        r"why", r"explain", r"reason", r"how did you", r"what made you",
        r"basis", r"evidence", r"justify",
    ],
    "quality": [
        r"quality", r"image.*good", r"usable", r"gradable",
        r"can you see", r"clear",
    ],
    "amd": [
        r"amd", r"macular degen", r"macula", r"drusen",
    ],
    "glaucoma": [
        r"glaucoma", r"cup.disc", r"cdr", r"optic disc", r"iop",
    ],
    "confidence": [
        r"confidence", r"certain", r"sure", r"accurate", r"reliable",
        r"trust", r"how sure",
    ],
    "progression": [
        r"worse", r"better", r"progress", r"change", r"compari",
        r"last visit", r"previous", r"trend", r"evolv",
    ],
    "summary": [
        r"summary", r"overview", r"tell me about", r"what.*this",
        r"overall", r"general", r"brief",
    ],
}


def classify_intent(question: str) -> List[str]:
    """Return list of matched intents for a question."""
    q = question.lower()
    matched = []
    for intent, patterns in INTENT_PATTERNS.items():
        if any(re.search(p, q) for p in patterns):
            matched.append(intent)
    return matched if matched else ["summary"]


# ─────────────────────────────────────────────────────────────────────────────
# Answer Generator
# ─────────────────────────────────────────────────────────────────────────────

DR_LABELS = {
    0: "No Diabetic Retinopathy",
    1: "Mild Non-Proliferative DR",
    2: "Moderate Non-Proliferative DR",
    3: "Severe Non-Proliferative DR",
    4: "Proliferative Diabetic Retinopathy",
}

LESION_NAMES = {
    "microaneurysm":     "microaneurysms",
    "hemorrhage":        "dot and blot hemorrhages",
    "hard_exudate":      "hard exudates",
    "soft_exudate":      "cotton-wool spots",
    "neovascularization":"neovascularization",
    "drusen":            "drusen deposits",
}


class RetinaCopilot:
    """
    AI Copilot that answers clinical questions about a retinal scan result.

    Usage:
        copilot = RetinaCopilot()
        answer  = copilot.ask("Should I refer this patient?", analysis_result)
    """

    def ask(self, question: str, result: Dict[str, Any],
            progression_report: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Answer a clinical question grounded in scan results.

        Args:
            question:           Doctor's natural language question
            result:             Full analysis result dict from /analyze
            progression_report: Optional progression data if available

        Returns:
            {
                "answer":     str — main answer text
                "confidence": float — [0,1] how well we can answer
                "intents":    list  — detected question categories
                "sources":    list  — which data points were used
                "suggestion": str   — follow-up suggestion
            }
        """
        intents = classify_intent(question)
        parts   = []
        sources = []

        for intent in intents:
            fragment, used = self._answer_intent(intent, result, progression_report)
            if fragment:
                parts.append(fragment)
                sources.extend(used)

        if not parts:
            parts, sources = self._answer_summary(result)

        answer     = " ".join(parts)
        confidence = self._estimate_confidence(result, intents)
        suggestion = self._suggest_followup(intents, result)

        return {
            "question":   question,
            "answer":     answer,
            "confidence": round(confidence, 2),
            "intents":    intents,
            "sources":    list(set(sources)),
            "suggestion": suggestion,
        }

    def _answer_intent(self, intent: str, r: Dict, prog: Optional[Dict]) -> Tuple[str, List[str]]:
        sources = []

        if intent == "summary":
            return " ".join(self._answer_summary(r)[0]), ["dr_grading","quality","lesions"]

        if intent == "dr_grade":
            dr = r.get("dr_grading", {})
            grade = dr.get("grade", 0)
            label = dr.get("label", DR_LABELS.get(grade, "Unknown"))
            conf  = round(dr.get("confidence", 0) * 100)
            sources.append("dr_grading")
            return (f"The AI grades this retina as {label} (Grade {grade}) "
                    f"with {conf}% confidence."), sources

        if intent == "refer":
            dr = r.get("dr_grading", {})
            glau = r.get("glaucoma", {})
            amd  = r.get("amd", {})
            refer = dr.get("refer", False)
            glau_s= glau.get("suspect", False)
            amd_l = amd.get("stage", 0)

            reasons = []
            if refer:         reasons.append(f"DR Grade {dr.get('grade',0)} ≥ 2 (referable threshold)")
            if glau_s:        reasons.append("glaucoma suspicion detected")
            if amd_l >= 3:    reasons.append("late AMD stage")

            sources.extend(["dr_grading","glaucoma","amd"])

            if reasons:
                return (f"Yes, referral is recommended. Reasons: {'; '.join(reasons)}. "
                        f"Urgency: {'urgent' if dr.get('grade',0) >= 4 else 'within 3 months'}."), sources
            else:
                return "No immediate referral required based on current findings. Routine annual screening is appropriate.", sources

        if intent == "lesions":
            lesions = r.get("lesions", {})
            present = {k: v for k, v in lesions.items() if v.get("present")}
            sources.append("lesions")
            if not present:
                return "No significant lesions were detected in this scan.", sources
            names = [f"{LESION_NAMES.get(k, k)} ({round(v['probability']*100)}%)" for k, v in present.items()]
            return f"The following lesions were detected: {', '.join(names)}.", sources

        if intent == "explain":
            dr     = r.get("dr_grading", {})
            lesions= r.get("lesions", {})
            grade  = dr.get("grade", 0)
            present= [LESION_NAMES.get(k, k) for k, v in lesions.items() if v.get("present")]
            sources.extend(["dr_grading","lesions","explainability"])

            if grade == 0:
                return "The scan shows a healthy retina with no signs of diabetic retinopathy. Vascular pattern appears normal.", sources
            elif grade == 1:
                return ("Mild DR was graded because microaneurysms were detected — "
                        "small red dots indicating early retinal vessel damage."), sources
            elif grade == 2:
                base = "Moderate DR was graded based on "
                findings = present if present else ["multiple retinal changes"]
                return base + f"{', '.join(findings)} visible in the fundus image.", sources
            elif grade >= 3:
                return ("Severe/Proliferative DR was detected. This indicates extensive retinal damage "
                        "with high risk of vision loss. Immediate ophthalmology evaluation required."), sources

        if intent == "quality":
            q = r.get("quality", {})
            score = round(q.get("score", 0) * 100)
            ok    = q.get("adequate", True)
            sources.append("quality")
            if ok:
                return f"Image quality is adequate for clinical analysis (quality score: {score}%).", sources
            else:
                return (f"Image quality is poor (score: {score}%). "
                        f"Results may be unreliable. Please retake the image with better illumination."), sources

        if intent == "amd":
            amd = r.get("amd", {})
            stage = amd.get("stage", 0)
            label = amd.get("label", "No AMD")
            conf  = round(amd.get("confidence", 0) * 100)
            sources.append("amd")
            if stage == 0:
                return f"No signs of Age-Related Macular Degeneration detected ({conf}% confidence).", sources
            else:
                return f"AMD findings: {label} (Stage {stage}, {conf}% confidence). Monitor macular region closely.", sources

        if intent == "glaucoma":
            g    = r.get("glaucoma", {})
            susp = g.get("suspect", False)
            cdr  = round(g.get("cup_disc_ratio", 0), 2)
            conf = round(g.get("confidence", 0) * 100)
            sources.append("glaucoma")
            if susp:
                return (f"Glaucoma suspicion detected. Cup-to-disc ratio is {cdr} "
                        f"(elevated, normal <0.6). Recommend IOP measurement and visual field testing."), sources
            else:
                return f"No glaucoma suspicion. Cup-to-disc ratio is {cdr} (within normal range).", sources

        if intent == "confidence":
            dr   = r.get("dr_grading", {})
            conf = round(dr.get("confidence", 0) * 100)
            probs= dr.get("probabilities", [])
            sources.append("dr_grading")
            top2_gap = 0
            if len(probs) >= 2:
                sp = sorted(probs, reverse=True)
                top2_gap = round((sp[0] - sp[1]) * 100)
            reliability = "high" if conf >= 85 else "moderate" if conf >= 70 else "low"
            return (f"Confidence is {conf}% ({reliability} reliability). "
                    f"The margin over the second-best prediction is {top2_gap}%."), sources

        if intent == "progression":
            if prog:
                trend = prog.get("overall_trend", "stable")
                risk  = round(prog.get("risk_12m", 0) * 100)
                change= prog.get("grade_change", 0)
                new_l = prog.get("new_lesions", [])
                sources.append("progression")
                direction = {"worsening": "worsened", "improving": "improved", "stable": "remained stable"}.get(trend, "changed")
                msg = f"Compared to previous visits, the condition has {direction}. "
                if change != 0: msg += f"DR grade changed by {'+' if change>0 else ''}{change}. "
                if new_l: msg += f"New lesions since baseline: {', '.join(new_l)}. "
                msg += f"12-month progression risk: {risk}%."
                return msg, sources
            else:
                return "No historical visits available for progression comparison. Analyze previous scans to enable this.", sources

        return "", []

    def _answer_summary(self, r: Dict) -> Tuple[List[str], List[str]]:
        dr    = r.get("dr_grading", {})
        grade = dr.get("grade", 0)
        conf  = round(dr.get("confidence", 0) * 100)
        refer = dr.get("refer", False)
        lesions = r.get("lesions", {})
        present = [LESION_NAMES.get(k, k) for k, v in lesions.items() if v.get("present")]

        parts = [
            f"This retinal scan shows {dr.get('label', DR_LABELS.get(grade, 'Unknown'))} "
            f"(Grade {grade}, {conf}% confidence)."
        ]

        if present:
            parts.append(f"Lesions detected: {', '.join(present)}.")
        else:
            parts.append("No significant lesions were identified.")

        if refer:
            parts.append("Ophthalmology referral is recommended.")
        else:
            parts.append("Routine follow-up is appropriate.")

        q = r.get("quality", {})
        if not q.get("adequate", True):
            parts.append("Note: Image quality is suboptimal — consider retaking.")

        return parts, ["dr_grading", "lesions", "quality"]

    def _estimate_confidence(self, r: Dict, intents: List[str]) -> float:
        base_conf = r.get("dr_grading", {}).get("confidence", 0.5)
        quality   = r.get("quality", {}).get("score", 0.5)
        return min(0.98, (base_conf * 0.7 + quality * 0.3))

    def _suggest_followup(self, intents: List[str], r: Dict) -> str:
        dr    = r.get("dr_grading", {})
        grade = dr.get("grade", 0)

        if "refer" in intents and grade >= 2:
            return "Ask: 'How urgently should I refer this patient?'"
        if "lesions" in intents:
            return "Ask: 'Explain why you graded this level of DR.'"
        if "explain" in intents:
            return "Ask: 'What follow-up tests do you recommend?'"
        if grade >= 3:
            return "Ask: 'What is the risk of vision loss without treatment?'"
        return "Ask: 'What is the recommended screening interval for this patient?'"


# Singleton
_copilot = RetinaCopilot()

def ask_copilot(question: str, result: Dict, progression: Optional[Dict] = None) -> Dict:
    return _copilot.ask(question, result, progression)
