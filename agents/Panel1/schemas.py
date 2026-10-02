"""Structured-output contracts for the cluster-selection agents (pydantic v2)."""
from __future__ import annotations

import os, sys
from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from agents.common.schemas_common import Strict, Citation   # noqa: E402  one citation contract, machine-verified


class Evidence(Strict):
    field: str = Field(description="dotted path of the packet field used, e.g. transport.posthoc_ad_to_ctrl.self_map_mean. Literature goes in citations, never here")
    value: str = Field(description="the value(s) read, as text")
    interpretation: str = Field(description="what this value licenses you to say, in one sentence")
    direction: Literal["supports_relevance", "against_relevance", "supports_suitability", "against_suitability", "neutral"]
    strength: Literal["strong", "moderate", "weak"]


class ContrastSide(Strict):
    group: Literal["case", "control", "both"] = Field(description="whose nuclei: the case donors, the control donors, or 'both' (only in an interaction test)")
    domains: List[str] = Field(description="unit ids whose nuclei form this side, e.g. ['domain_3']; several ids pool those domains")


class ProposedContrast(Strict):
    question: str = Field(description="what this comparison asks, <= 40 words")
    test: Literal["difference", "interaction"] = Field(description="difference: side_a vs side_b (each side one group); interaction: whether the side_a vs side_b difference differs between case and control (both sides group 'both')")
    side_a: ContrastSide = Field(description="the side of interest; log2FC is side_a relative to side_b")
    side_b: ContrastSide = Field(description="the reference side")
    cell_type: str = Field(description="'all' (cell type then enters the model as a covariate) or one prior cell type, the same on both sides")
    evidence_fields: List[str] = Field(description="dotted packet paths, prefixed with the unit, that motivate this comparison, e.g. domain_3.transport.posthoc_ad_to_ctrl.destination_domain_share_mean")
    rationale: str = Field(description="why the comparison is meaningful and what its result would mean, <= 80 words")
    main_confound: str = Field(description="what else could produce a difference in this comparison")
    prediction: str = Field(description="the outcome this comparison would show if your reading is right")


class UnitAssessment(Strict):
    unit: str
    relevance: float = Field(ge=0, le=1, description="0-1: how strongly disease acts in this unit, from this perspective only")
    suitability: float = Field(ge=0, le=1, description="0-1: how cleanly a case vs control contrast can be estimated here")
    confidence: float = Field(ge=0, le=1)
    justification: str = Field(description="<= 120 words citing packet fields")
    evidence: List[Evidence]
    main_confound: str = Field(description="the single alternative explanation most threatening to your reading")
    falsifiable_prediction: str = Field(description="one concrete outcome DEG or a follow-up would show if your reading is right")
    recommended_strata: List[str] = Field(description="cell types within this domain worth testing separately in the default contrast (this domain, case vs control), e.g. ['Ast', 'End']; empty if none")
    proposed_contrasts: List[ProposedContrast] = Field(description="comparisons beyond the default that your evidence gives a reason to run, with this unit on side_a; empty if none")
    citations: List[Citation] = Field(description="literature behind any mechanism claim in this assessment, from web searches run in this session; machine-verified against those search results and PubMed before the critic reads it. Empty if you make no literature claim")


class AgentReport(Strict):
    agent: str
    assessments: List[UnitAssessment]
    cross_unit_notes: str = Field(description="patterns across units visible only from this perspective; <= 150 words")


class EvidenceVerdict(Strict):
    agent: str
    unit: str
    field: str
    verdict: Literal["independent", "redundant", "explained_away", "unverifiable"]
    reason: str


class CriticUnit(Strict):
    unit: str
    verdicts: List[EvidenceVerdict]
    independent_lines_for_relevance: int = Field(ge=0, description="count of distinct, surviving evidence lines for relevance")
    independent_lines_against: int = Field(ge=0)
    suitability_pass: bool
    notes: str


class ContrastVerdict(Strict):
    proposal_id: str
    verdict: Literal["sound", "unsound"]
    reason: str = Field(description="the packet fields and reasoning behind the verdict")


class CriticReport(Strict):
    units: List[CriticUnit]
    contrast_verdicts: List[ContrastVerdict] = Field(description="one verdict per entry of contrast_proposals")
    general_notes: str


class RankedUnit(Strict):
    unit: str
    rank: int
    decision: Literal["shortlist", "hold", "exclude"]
    relevance_consensus: float = Field(ge=0, le=1)
    suitability_consensus: float = Field(ge=0, le=1)
    independent_lines: int
    rationale: str = Field(description="<= 100 words; name the perspectives that agree and the confound that was considered")
    strata: List[str]
    predictions: List[str] = Field(description="the falsifiable predictions carried forward as pre-registered hypotheses")


class AdoptedContrast(Strict):
    from_proposals: List[str] = Field(description="ids of the proposals this contrast adopts; duplicates of one comparison are merged into one entry")
    question: str
    test: Literal["difference", "interaction"]
    side_a: ContrastSide
    side_b: ContrastSide
    cell_type: str
    main_confound: str
    prediction: str = Field(description="carried forward verbatim from the proposal(s)")


class SynthesisReport(Strict):
    rule_applied: str
    ranking: List[RankedUnit]
    contrasts: List[AdoptedContrast] = Field(description="the additional contrasts adopted under the contrast rule; empty if none qualifies")
    caveats: str
