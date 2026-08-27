from training.data_pipeline.terms import annotate_medical_terms


def test_term_annotation_does_not_rewrite_transcript(fixture_config) -> None:
    records = [
        {
            "normalized_transcript": "Take 5 mg aspirin for chest pain and get an MRI",
        }
    ]
    original = records[0]["normalized_transcript"]

    annotate_medical_terms(records, fixture_config)

    assert records[0]["normalized_transcript"] == original
    categories = set(records[0]["medical_term_categories"])
    assert {
        "dosage",
        "medication",
        "symptom",
        "procedure",
        "abbreviation",
    } <= categories
    assert all(
        annotation["text"] == original[annotation["start"] : annotation["end"]]
        for annotation in records[0]["medical_terms"]
    )
