"""
Structured-output contracts for the TARGET panel (pydantic v2).

Generic by construction: no field here names a disease, tissue, organism, gene
or programme. Everything specific arrives in the packets and the dataset notes,
so the same contracts serve any dataset the upstream stages ran on.

Two contracts encode decisions that would otherwise be made silently:

  * DirectionAssessment separates `observed_expression_direction` from
    `mechanism_direction` and forces a `resolution` when they disagree. The
    direction a target should be modulated in is NOT readable off its expression:
    a negative regulator of a process that is being lost should be INHIBITED to
    restore that process, whatever its own expression does. Collapsing the two
    into one field is the failure mode this schema exists to prevent.

Chemistry is deliberately absent from these contracts. Whether bioactivity data
exists gates nothing at target-selection time. No field
here can be used to reject a target for want of compounds.

`evidence_status` appears wherever a judgement could be depressed by an absence
of literature. "no_evidence_found" and "evidence_of_no_effect" are different
findings, and an under-studied target must record the former rather than be
scored as though the latter were true.
"""
from __future__ import annotations

import os, sys
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from agents.common.schemas_common import Strict, Citation, EvidenceStatus   # noqa: E402,F401


class Evidence(Strict):
    field: str = Field(description="dotted path of the packet field used, e.g. axis3_topology.networks.<programme>.scales_top. Literature goes in citations, never here")
    value: str = Field(description="the value(s) read, as text")
    interpretation: str = Field(description="what this value licenses you to say, in one sentence")
    strength: Literal["strong", "moderate", "weak"]


# --------------------------------------------------------------------------- #
# Axis specialists: one axis each, no knowledge of the others
# --------------------------------------------------------------------------- #
class AxisAssessment(Strict):
    gene: str
    domain: str
    score: float = Field(ge=0, le=1, description="0-1: how strongly YOUR axis alone implicates this gene as a target here")
    confidence: float = Field(ge=0, le=1)
    claim: str = Field(description="<= 60 words: the specific claim your axis supports for this gene")
    not_licensed: str = Field(description="<= 40 words: what a reader must NOT conclude from your axis about this gene")
    evidence: List[Evidence]
    main_alternative: str = Field(description="the reading of these numbers that would make this gene uninteresting")


class AxisReport(Strict):
    agent: str
    assessments: List[AxisAssessment]
    cross_candidate_notes: str = Field(description="patterns across candidates visible only from this axis; <= 150 words")


# --------------------------------------------------------------------------- #
# Direction: inhibit or activate
# --------------------------------------------------------------------------- #
class DirectionAssessment(Strict):
    gene: str
    domain: str
    target_process: str = Field(description="the process to be corrected, taken from the packet's process_direction block, and which way it moves in disease")
    gene_valence: Literal["positive_regulator", "negative_regulator", "structural_component",
                          "effector_downstream", "unknown"] = Field(
        description="the gene's role WITH RESPECT TO that process, from mechanism, not from its own expression")
    valence_evidence: EvidenceStatus
    observed_expression_direction: Optional[Literal["up_in_disease", "down_in_disease", "not_measured"]] = Field(None, 
        description="what this dataset says the gene's own expression does; a CHECK, never the basis of the recommendation")
    mechanism_direction: Literal["inhibit", "activate", "undetermined"] = Field(
        description="compose target_process with gene_valence: restoring a lost process means ACTIVATING its positive regulators and INHIBITING its negative regulators")
    agrees_with_expression: bool
    resolution: str = Field(description="<= 80 words. If mechanism and expression disagree, say why mechanism is still preferred (commonly a compensatory response) or why this case is genuinely undetermined")
    citations: List[Citation] = Field(
        description="references supporting the valence and direction. Empty list is the correct "
                    "answer when you are reasoning from general domain knowledge with no specific "
                    "source in mind - say so rather than attaching a reference you are unsure of. "
                    "An empty list with valence_evidence 'suggestive' is honest; a fabricated "
                    "citation is a fatal error")
    essentiality_risk: Literal["low", "moderate", "high", "unknown"] = Field(
        description="high for broadly required housekeeping functions where modulation either way is likely toxic")
    expected_harm: str = Field(description="<= 60 words: the most plausible way modulating this target in the stated direction makes things worse")
    confidence: float = Field(ge=0, le=1)
    evidence: List[Evidence]


class DirectionReport(Strict):
    agent: str
    assessments: List[DirectionAssessment]
    notes: str


# --------------------------------------------------------------------------- #
# Critic and synthesis
# --------------------------------------------------------------------------- #
class EvidenceVerdict(Strict):
    agent: str
    gene: str
    domain: str
    field: str
    verdict: Literal["independent", "redundant", "explained_away", "unverifiable"]
    reason: str


class CriticCandidate(Strict):
    gene: str
    domain: str
    verdicts: List[EvidenceVerdict]
    independent_lines: int = Field(ge=0, description="number of DATA axes (dysregulation, programme, topology) with at least one item labelled independent; literature items never count")
    direction_is_mechanism_grounded: bool = Field(description="false if the direction rests on expression sign alone with no pathway or literature basis")
    novelty_penalty_detected: bool = Field(description="true if any agent scored this candidate down for being under-studied rather than for evidence against it")
    disqualifying_concern: Optional[str] = Field(None, description="a concern that should remove this candidate regardless of scores, e.g. the signal tracks a known technical artefact, or the protein is broadly essential; null if none")
    notes: str


class TargetCriticReport(Strict):
    candidates: List[CriticCandidate]
    general_notes: str


class RankedTarget(Strict):
    gene: str
    domains: List[str] = Field(description="every domain in which this gene was assessed and survived")
    rank: int
    tier: Literal["convergent", "single_axis_strong_mechanism", "hold", "rejected"]
    biological_case: float = Field(ge=0, le=1, description="evidence that modulating this gene should matter. Chemical tractability is assessed at the repurposing stage and must NOT enter this number")
    direction: Literal["inhibit", "activate", "undetermined"]
    direction_basis: str = Field(description="<= 50 words: the mechanism that sets the direction, and whether expression agreed")
    supporting_axes: List[str] = Field(description="data axes only: dysregulation, programme, topology")
    rationale: str = Field(description="<= 100 words; name the axes that agree and the alternative that was considered")
    rejection_reason: Optional[str] = Field(None, description="required when tier is 'rejected'; must not be 'under-studied' alone")
    next_experiment: str = Field(description="one concrete experiment that would most change confidence in this target")


class TargetSynthesisReport(Strict):
    rule_applied: str = Field(description="state the rule you applied, including how single-axis candidates and absent chemistry were handled")
    ranking: List[RankedTarget]
    novel_candidates: List[str] = Field(description="targets whose case rests on this analysis's data with little or no prior literature, including those held only because no direction could be sourced")
    caveats: str
