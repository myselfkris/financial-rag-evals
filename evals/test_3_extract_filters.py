"""
Unit test for Step 1 of the filter retriever: reading company + year(s).
Hand-written questions, NOT taken from the frozen test set.

Run:  python -m pytest evals/test_3_extract_filters.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.retriever_8_filter_fix import extract_filters

CASES = [
    # question, expected companies, expected years, what it checks
    ("What was AMCOR's revenue in FY22?", ["Amcor"], ["2022"], "all-caps name + 2-digit year"),
    ("How did AMD's margin change from FY21 to FY22?", ["AMD"], ["2021", "2022"], "two 2-digit years"),
    ("What was 3M's capex in FY2018?", ["3M"], ["2018"], "already worked; must not break"),
    ("What is Amcor's main industry?", ["Amcor"], [], "no year -> don't invent one"),
    ("Did Adobe's revenue grow 22% in FY2016?", ["Adobe"], ["2016"], "'22%' is not a year"),
    ("Does AMEX have an improving operating margin in 2022?", ["American Express"], ["2022"], "nickname"),
    ("Did 3M sell a block of shares in 2018?", ["3M"], ["2018"], "word 'block' is not the company Block"),
]


@pytest.mark.parametrize("question,companies,years,checks", CASES, ids=[c[3] for c in CASES])
def test_extract_filters(question, companies, years, checks):
    got = extract_filters(question)
    assert got["companies"] == companies
    assert got["years"] == years
