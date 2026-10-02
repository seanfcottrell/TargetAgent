"""Structured-output contract for the report writer (pydantic v2)."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FinalReport(Strict):
    title: Optional[str] = Field(None, description="the report's title; the model sometimes omits it, so it is optional")
    executive_summary: str = Field(description=(
        "<= 300 words: what was measured, what the two panels landed on (domains, targets), and "
        "the caveats that most limit them. No proposed work."))
    data_and_cohort: str = Field(description=(
        "tissue, technology, cohort, batch structure, QC by group, and how nuclei were typed "
        "with the composition found"))
    spatial_domains: str = Field(description=(
        "what the domains ARE: each domain described by its measured composition, its programmes "
        "and their genes and terms, and the case-control shifts measured in it"))
    domain_panel: str = Field(description=(
        "the domain panel's output: which domains it shortlisted and which it held or excluded, "
        "its rule, the evidence lines it upheld or ruled out, and what the shortlisted domains are"))
    measured_evidence: str = Field(description=(
        "the two computed evidence axes behind the targets: the differential expression contrasts "
        "run and their hits and recurrence, and what the topology axis selected per domain"))
    target_panel: str = Field(description=(
        "the target panel's output and the mechanical selection on data merit: what each selected "
        "target IS, the axes supporting it, directions and direction-open targets, exclusions and "
        "the reasons, and the panel's tiers where they differ from the data tiers"))
    biological_narrative: str = Field(description=(
        "whether the domains, programmes and targets converge on one coherent "
        "biological story or several, built only from what was measured and what the panels "
        "recorded; name explicitly where they do NOT converge or where a result stands alone"))
    caveats_and_limitations: str
