from app.defaults.data_synth import generate_synthetic_dataset
from app.io.feature_loader import get_model_feature_order


def test_synthetic_dataset_includes_configured_other_sex_cohort() -> None:
    features, _ = generate_synthetic_dataset(n=20_000, random_state=42)

    proportions = features["sex_at_birth"].value_counts(normalize=True)
    assert set(proportions.index) == {0, 1, 2}
    assert 0.015 < proportions[2] < 0.025
    assert list(features.columns) == get_model_feature_order()
