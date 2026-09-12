import os
import time
import requests
from dotenv import load_dotenv
from feature_mapping import clean_feature_name

# --------------------------------------------------
# 1. Load API key
# --------------------------------------------------

load_dotenv()

AMASS_API_KEY = os.getenv("AMASS_API_KEY")

BASE_URL = "https://api.amass.tech/api/v1"

HEADERS = {
    "Authorization": f"Bearer {AMASS_API_KEY}"
}


# --------------------------------------------------
# 2. Generic Amass GET request
# --------------------------------------------------

def fetch_with_retry(url, params=None, retries=3):
    """
    Send a GET request to Amass.

    If Amass rate-limits us with HTTP 429,
    wait and retry automatically.
    """

    if not AMASS_API_KEY:
        raise ValueError(
            "AMASS_API_KEY was not found. "
            "Check your .env file."
        )

    for attempt in range(retries):

        response = requests.get(
            url,
            headers=HEADERS,
            params=params,
            timeout=30
        )

        # Rate limit
        if response.status_code == 429:

            wait_time = int(
                response.headers.get(
                    "Retry-After",
                    2 ** attempt
                )
            )

            print(
                f"Rate limited by Amass. "
                f"Waiting {wait_time} seconds..."
            )

            time.sleep(wait_time)
            continue

        # Raise error for 400/401/403/500 etc.
        response.raise_for_status()

        result = response.json()

        return result.get("data", [])

    return []

def fetch_biomed_evidence(factor: str):
    """
    Search BiomedCore for scientific papers related
    to a brain-aging factor.
    """

    query = f"{factor} brain aging"

    print(f"Searching BiomedCore for: {query}")

    papers = fetch_with_retry(
        f"{BASE_URL}/cores/biomedcore/records",
        params={
            "query": query,
            "minJournalQualityJufo": 1,
            "minCitationCount": 5,
            "limit": 5
        }
    )

    return papers

import re


AGING_TEXT_TERMS = [
    "brain aging",
    "healthy aging",
    "normal aging",
    "age-related",
    "age associated",
    "age-associated",
    "older adults",
    "older adult",
    "elderly",
    "lifespan",
]


GENERIC_FACTOR_WORDS = {
    "right",
    "left",
    "brain",
    "cerebral",
    "cortex",
    "cortical",
    "volume",
    "thickness",
    "surface",
    "area",
    "mean",
    "total",
    "matter",
}


def _normalize(text):
    """
    Lowercase text and normalize punctuation/spaces.
    """

    if not text:
        return ""

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text
    )

    return text.strip()


def _factor_keywords(factor):
    """
    Extract the anatomically important words
    from a mapped feature name.
    """

    words = _normalize(factor).split()

    return [
        word
        for word in words
        if word not in GENERIC_FACTOR_WORDS
        and len(word) > 2
    ]


def score_biomed_paper(factor, paper):
    """
    Give an Amass paper a simple relevance score.

    We care about:
    1. aging context
    2. anatomical feature match
    3. paper quality
    """

    title = _normalize(
        paper.get("title", "")
    )

    abstract = _normalize(
        paper.get("abstract", "")
    )

    text = f"{title} {abstract}"

    mesh_terms = [
        _normalize(term)
        for term in paper.get("meshTerms", [])
    ]

    keywords = [
        _normalize(term)
        for term in paper.get("keywords", [])
    ]

    score = 0


    # ----------------------------------------------
    # Aging relevance
    # ----------------------------------------------

    aging_match = any(
        _normalize(term) in text
        for term in AGING_TEXT_TERMS
    )

    if aging_match:
        score += 4


    # MeSH metadata is stronger evidence
    aging_mesh_terms = {
        "aging",
        "age factors",
        "aged",
        "middle aged",
    }

    if any(
        term in aging_mesh_terms
        for term in mesh_terms
    ):
        score += 3


    # ----------------------------------------------
    # Anatomical relevance
    # ----------------------------------------------

    factor_words = _factor_keywords(
        factor
    )

    title_hits = sum(
        word in title
        for word in factor_words
    )

    abstract_hits = sum(
        word in abstract
        for word in factor_words
    )

    score += min(title_hits, 2) * 2
    score += min(abstract_hits, 2)


    # ----------------------------------------------
    # Scientific quality signals
    # ----------------------------------------------

    journal_quality = (
        paper.get("journalQualityJufo")
        or 0
    )

    citations = (
        paper.get("citationCount")
        or 0
    )

    if journal_quality >= 2:
        score += 1

    if citations >= 20:
        score += 1


    # Never use retracted work
    if paper.get("isRetracted"):
        score = -100


    return score


def filter_biomed_evidence(
    factor,
    papers,
    top_k=3,
    min_score=6
):
    """
    Rank Amass papers and retain only reasonably
    relevant evidence.
    """

    scored = []

    for paper in papers:

        score = score_biomed_paper(
            factor,
            paper
        )

        if score >= min_score:

            paper_copy = paper.copy()

            paper_copy[
                "relevanceScore"
            ] = score

            scored.append(
                paper_copy
            )


    scored.sort(
        key=lambda paper: (
            paper["relevanceScore"],
            paper.get("citationCount", 0)
        ),
        reverse=True
    )

    return scored[:top_k]

def fetch_trial_evidence(factor: str):
    """
    Search clinical trials related to a brain-aging factor.
    """

    query = f"{factor} brain"

    print(f"Searching TrialCore for: {query}")

    trials = fetch_with_retry(
        f"{BASE_URL}/cores/trialcore/records",
        params={
            "query": query,
            "studyType": "INTERVENTIONAL",
            "limit": 5,
            "include": "outcomes"
        }
    )

    return trials

if __name__ == "__main__":

    raw_feature = "ThirdVentVol"

    factor = clean_feature_name(raw_feature)

    print("Raw model feature:", raw_feature)
    print("Medical search term:", factor)

    trials = fetch_trial_evidence(factor)

    print("\nNumber of trials found:", len(trials))

    for trial in trials:
        print("\nTitle:", trial.get("briefTitle"))
        print("Conditions:", trial.get("conditions"))
        print("Status:", trial.get("overallStatus"))
        print("NCT ID:", trial.get("nctId"))
        print("-" * 50)