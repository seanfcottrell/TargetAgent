"""
Structured-output pieces shared by every panel (pydantic v2): the strict base
model and the ONE citation contract, machine-verified by agents/common/citations.py.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


EvidenceStatus = Literal["well_established", "suggestive", "no_evidence_found", "evidence_of_no_effect"]




class Citation(Strict):
    """A literature reference, in enough detail to be checked.

    Free-text literature claims proved unfalsifiable in practice: an agent can
    assert a mechanism with no reference at all and it reads exactly like a
    supported finding. Requiring the parts of a real citation makes an
    unsupported claim visible as one, and makes a fabricated one checkable
    against PubMed. Partial is allowed and honest; inventing is not.

    Every citation is MACHINE-VERIFIED (agents/citations.py): it must match a
    result of a web search run during this panel, and a PMID must resolve in
    PubMed to the same title, first author and year. Recalled references do
    not count as support even when they exist.
    """
    claim: str = Field(description="the specific finding this reference supports, in one sentence")
    title: Optional[str] = Field(None, description="article title exactly as the search result showed it; null if you have no search result")
    first_author: Optional[str] = Field(None, description="first author surname as shown in the source; null if not shown")
    year: Optional[int] = Field(None, description="publication year as shown in the source; null if not shown")
    pmid: Optional[str] = Field(None, description="PubMed ID, digits only, ONLY if a search result showed it (e.g. in a pubmed.ncbi.nlm.nih.gov URL). Never from memory: every PMID is checked against PubMed and a wrong one is recorded as a fabrication")
    url: Optional[str] = Field(None, description="the URL of the search result this came from, exactly as returned; null if none")
    source: Literal["web_search_result", "model_knowledge"] = Field(
        description="'web_search_result' only if a search in THIS session returned it; "
                    "'model_knowledge' marks a recalled reference, which does not count as support")
    confidence_in_reference: Literal["verified_this_session", "confident", "uncertain"]
