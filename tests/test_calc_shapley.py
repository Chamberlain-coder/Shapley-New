import importlib.util
import math
from pathlib import Path


def load_calc_shapley_module():
    module_path = Path(__file__).resolve().parents[1] / "analysis" / "calc_shapley.py"
    spec = importlib.util.spec_from_file_location("calc_shapley", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_data_supports_legacy_and_gating_formats():
    calc_shapley = load_calc_shapley_module()

    parsed = calc_shapley.parse_data(
        {
            "(0, 1, 2, 3)": 5,
            "(1, 2, 3, 4)": {
                "count": 2,
                "gating_score": [0.1, 0.2, 0.3, 0.6],
            },
        }
    )

    assert parsed[(0, 1, 2, 3)]["count"] == 5
    assert parsed[(0, 1, 2, 3)]["value"] == 5.0
    assert parsed[(1, 2, 3, 4)]["count"] == 2
    assert parsed[(1, 2, 3, 4)]["value"] == 1.2
    assert parsed[(1, 2, 3, 4)]["expert_scores"][4] == 0.6


def test_process_layer_uses_gating_score_sum_for_shapley():
    calc_shapley = load_calc_shapley_module()

    legacy_layer_df = calc_shapley.process_layer(
        "0",
        {
            "(0, 1, 2, 3)": 100,
            "(0, 1, 2, 4)": 1,
            "(0, 2, 3, 5)": 100,
            "(0, 2, 3, 4)": 1,
        },
    )
    layer_df = calc_shapley.process_layer(
        "0",
        {
            "(0, 1, 2, 3)": {"count": 100, "gating_score": [0.25, 0.25, 0.25, 0.25]},
            "(0, 1, 2, 4)": {"count": 1, "gating_score": [0.2, 0.3, 0.4, 3.0]},
            "(0, 2, 3, 5)": {"count": 100, "gating_score": [0.25, 0.25, 0.25, 0.25]},
            "(0, 2, 3, 4)": {"count": 1, "gating_score": [0.2, 0.3, 0.4, 5.0]},
        },
    )

    assert layer_df is not None
    assert legacy_layer_df is not None

    expert_4 = layer_df[layer_df["Expert_ID"] == 4].iloc[0]
    legacy_expert_4 = legacy_layer_df[legacy_layer_df["Expert_ID"] == 4].iloc[0]

    assert expert_4["Total_Activations"] == 2
    assert math.isclose(expert_4["Total_Gating_Score"], 8.0)
    assert math.isclose(expert_4["Avg_Gating_Score"], 4.0)
    assert expert_4["Shapley_Value"] > 0
    assert not math.isclose(
        expert_4["Shapley_Value"],
        legacy_expert_4["Shapley_Value"],
    )
