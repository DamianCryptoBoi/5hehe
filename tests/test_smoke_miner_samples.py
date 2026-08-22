"""Sample miner smoke-test input isolation."""

from scripts.smoke_miner_samples import _request_for_sample, load_sample


def test_problem_only_request_contains_no_authored_cases():
    payload, statement, cases, _labels = load_sample("extent-journal")

    request = _request_for_sample(
        "extent-journal",
        payload,
        statement,
        cases,
        300.0,
        problem_only=True,
    )

    assert request.statement == statement
    assert request.public_examples == []


def test_default_sample_request_still_contains_one_public_case():
    payload, statement, cases, _labels = load_sample("extent-journal")

    request = _request_for_sample(
        "extent-journal",
        payload,
        statement,
        cases,
        300.0,
    )

    assert request.public_examples == [cases[0]]
