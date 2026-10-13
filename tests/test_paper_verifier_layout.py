"""Table 7 metadata checks, not substitute implementations of its predicates."""

from dataclasses import replace

import pytest

from vapa.schemas import ActionKind
from vapa.verifiers import VerifierCatalog, VerifierFamily, paper_predicate_layout


def supplied_functions():
    # Only a binding fixture: never used as the executable paper catalog.
    return {
        row["name"]: lambda context, trace, index: None
        for row in paper_predicate_layout()["predicates"]
    }


def test_table_seven_weights_action_normalizers_and_source():
    layout = paper_predicate_layout()
    assert layout["source"] == {
        "manuscript_sha256": "785d29fd7437824a497d72b76b5b89ddd4f3853ae5d8e6bdc1056e1f09dbb64d",
        "section": "Appendix A.2",
        "table": 7,
        "page": 19,
    }
    assert layout["normalizers"] == {
        "RETRIEVE": 1,
        "QUERYFIELD": 5,
        "CALCULATE": 13,
        "UPDATEMEMORY": 11,
        "MARKSTATUS": 2,
        "COMPRESS": 2,
        "DISCARD": 2,
        "ANSWER": 16,
    }
    assert set(layout["normalizers"]) == {
        action.value for action in ActionKind if action is not ActionKind.MALFORMED
    }
    assert len(layout["predicates"]) == 14
    assert sum(row["weight"] == 2 for row in layout["predicates"]) == 6
    assert sum(row["weight"] == 1 for row in layout["predicates"]) == 8
    assert layout["score_range"] == [-1, 1]
    assert layout["abstention"] is None


def test_binding_does_not_claim_verified_semantics():
    functions = supplied_functions()
    catalog = VerifierCatalog.from_paper_functions(functions, catalog_id="binding-fixture")
    assert not catalog.paper_exact
    catalog.validate_paper_layout()
    assert all(predicate.function is functions[predicate.name] for predicate in catalog.predicates)


@pytest.mark.parametrize("change", ["missing", "extra", "noncallable"])
def test_binding_rejects_incomplete_or_invalid_implementations(change):
    functions = supplied_functions()
    if change == "missing":
        functions.pop("ExactExtraction")
    elif change == "extra":
        functions["Guess"] = lambda context, trace, index: None
    else:
        functions["ExactExtraction"] = None
    with pytest.raises(ValueError):
        VerifierCatalog.from_paper_functions(functions, catalog_id="invalid")


@pytest.mark.parametrize(
    "change",
    [
        {"name": "wrong"},
        {"weight": 1},
        {"family": VerifierFamily.STOPPING},
        {"actions": frozenset({ActionKind.ANSWER})},
        {"reliability": "heuristic"},
    ],
)
def test_paper_exact_flag_rejects_wrong_table_assignment(change):
    catalog = VerifierCatalog.from_paper_functions(supplied_functions(), catalog_id="fixture")
    predicates = (replace(catalog.predicates[0], **change), *catalog.predicates[1:])
    with pytest.raises(ValueError, match="Table 7"):
        VerifierCatalog(predicates, catalog_id="invalid", paper_exact=True)
