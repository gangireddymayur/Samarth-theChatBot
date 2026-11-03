import pytest

from conftest import ask


def test_datasets_are_ingested(client):
    tables = client.get("/datasets").json()
    assert {"ds_rainfall_subdiv", "ds_crop_production"} <= set(tables)
    assert all(meta["nrows"] > 0 for meta in tables.values())


@pytest.mark.parametrize(
    "query, expected_in_answer",
    [
        ("Compare average annual rainfall in Kerala and Tamil Nadu for the last 5 years", ["Kerala", "Tamil Nadu"]),
        ("Trend of rainfall in Kerala for the last 10 years", ["Kerala", "2008-2017"]),
        ("Which subdivision had the lowest rainfall in 2015?", ["lowest", "2015"]),
        ("Top 5 most produced crops in Maharashtra for the last 5 years", ["Maharashtra", "Cotton"]),
        ("Analyze the trend of Rice production in Andhra Pradesh for the last 15 years", ["Andhra Pradesh"]),
        ("Compare production in Punjab and Haryana for the last 5 years", ["Punjab", "Haryana"]),
        ("Correlate rice production in Andhra Pradesh for the last 10 years with rainfall", ["Pearson", "rice"]),
        ("Compare rainfall and top crops in Maharashtra and Karnataka for the last 5 years", ["Maharashtra", "Karnataka"]),
    ],
)
def test_supported_intents_answer_with_sql_evidence(client, query, expected_in_answer):
    resp = ask(client, query)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    for text in expected_in_answer:
        assert text in body["answer"]
    # Every answer must be traceable to the SQL that produced it.
    assert "SELECT" in str(body["evidence"]).upper()


def test_last_n_years_window_is_respected(client):
    body = ask(client, "Top 5 most produced crops in Maharashtra for the last 5 years").json()
    assert "(2016-2020)" in body["answer"]


def test_unknown_question_is_rejected_with_examples(client):
    resp = ask(client, "What is the capital of France?")
    assert resp.status_code == 400
    assert "Could not parse query" in resp.json()["detail"]


def test_empty_question_is_rejected(client):
    resp = ask(client, "   ")
    assert resp.status_code == 400
    assert resp.json()["detail"] == "Empty query provided."


@pytest.mark.xfail(reason="State names containing 'and' are split on the 'and' separator", strict=True)
def test_state_name_containing_and(client):
    body = ask(client, "Top 3 crops in Andaman and Nicobar Islands").json()
    assert "Andaman and Nicobar Islands" in body["answer"]
