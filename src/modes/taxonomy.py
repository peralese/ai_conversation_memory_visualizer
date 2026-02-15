from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModeRuleSet:
    id: str
    name: str
    description: str
    strong_keywords: tuple[str, ...]
    medium_keywords: tuple[str, ...]
    weak_keywords: tuple[str, ...]


MODE_TAXONOMY: tuple[ModeRuleSet, ...] = (
    ModeRuleSet(
        id="analytical_reasoning",
        name="Analytical Reasoning",
        description="Evaluating tradeoffs, evidence, constraints, and impacts.",
        strong_keywords=(
            "analyze",
            "analysis",
            "tradeoff",
            "pros",
            "cons",
            "impact",
            "compare",
            "evaluate",
            "hypothesis",
            "evidence",
        ),
        medium_keywords=("rationale", "metric", "risk", "assumption", "constraint"),
        weak_keywords=("why", "assess", "consider"),
    ),
    ModeRuleSet(
        id="design_synthesis",
        name="Design & Synthesis",
        description="Designing systems, architectures, and structured approaches.",
        strong_keywords=("design", "architecture", "model", "framework", "blueprint", "structure", "system", "approach"),
        medium_keywords=("diagram", "components", "integration", "pipeline", "schema"),
        weak_keywords=("pattern", "layout"),
    ),
    ModeRuleSet(
        id="decision_framing",
        name="Decision Framing",
        description="Framing options and selecting among alternatives.",
        strong_keywords=("decide", "decision", "recommend", "option", "choice", "alternative", "select", "best", "should we"),
        medium_keywords=("priority", "criteria", "ranking", "guidance"),
        weak_keywords=("suggest",),
    ),
    ModeRuleSet(
        id="communication_refinement",
        name="Communication & Refinement",
        description="Rewriting and refining communication for clarity and audience.",
        strong_keywords=("rewrite", "edit", "polish", "improve", "simplify", "clarify", "summarize", "email", "deck", "slide", "executive"),
        medium_keywords=("wording", "tone", "concise", "narrative", "headline"),
        weak_keywords=("message", "explain"),
    ),
    ModeRuleSet(
        id="troubleshooting_debugging",
        name="Troubleshooting & Debugging",
        description="Diagnosing failures and fixing issues.",
        strong_keywords=("error", "failing", "issue", "bug", "troubleshoot", "diagnose", "fix", "not working", "stack trace"),
        medium_keywords=("investigate", "reproduce", "logs", "latency", "timeout"),
        weak_keywords=("problem",),
    ),
    ModeRuleSet(
        id="learning_research",
        name="Learning & Research",
        description="Explaining concepts and exploring knowledge.",
        strong_keywords=("what is", "explain", "how does", "difference between", "learn", "tutorial", "guide", "overview"),
        medium_keywords=("example", "basics", "concept", "definition"),
        weak_keywords=("intro",),
    ),
    ModeRuleSet(
        id="planning_organization",
        name="Planning & Organization",
        description="Roadmaps, checklists, sequencing, and prioritization.",
        strong_keywords=("plan", "roadmap", "milestone", "next steps", "checklist", "schedule", "timeline", "prioritize"),
        medium_keywords=("backlog", "tasks", "phases", "deliverables"),
        weak_keywords=("organize",),
    ),
    ModeRuleSet(
        id="creative_generative",
        name="Creative & Generative Work",
        description="Idea generation, creative variants, and exploratory outputs.",
        strong_keywords=("brainstorm", "ideas", "generate", "creative", "concept", "story", "examples", "variations"),
        medium_keywords=("inspiration", "imagine", "draft concepts"),
        weak_keywords=("try",),
    ),
)


MODE_BY_ID = {m.id: m for m in MODE_TAXONOMY}
