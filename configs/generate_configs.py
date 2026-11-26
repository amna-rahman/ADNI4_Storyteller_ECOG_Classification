"""Generate configs/baseline.yaml and configs/proposed_model.yaml for the ADNI4 project.

Run this script from your main project folder (the folder containing data/, src/, results/, figures/):
    python generate_configs.py
"""

from pathlib import Path
import yaml


BASELINE_CONFIG = {
    "experiment_group": "baseline",
    "project": {
        "name": "ADNI4 Speech and Participant Self-Reported Everyday Cognition (ECog-12 PT) MCI Detection",
        "ecog_source_table": "ECOG12PT (Everyday Cognition 12 Item - Participant Self-Report)",
        "primary_cohort_unit": "assessment",
        "evaluation_unit": "participant",
        "target_encoding": {"CN": 0, "MCI": 1},
    },
    "paths": {
        "raw_data_dir": "data/raw",
        "processed_data_dir": "data/processed",
        "notebooks_dir": "src",
        "results_root": "results",
        "figures_root": "figures",
        "datasets": {
            "storyteller_source": "data/processed/dataset1_storyteller.csv",
            "ecog_source": "data/processed/dataset2_ecog.csv",
            "primary_multimodal_30d": "data/processed/primary_multimodal_30day_cohort.csv",
            "all_pairs_extension": "data/processed/dataset4_multimodal_cn_vs_noncn.csv",
        },
    },
    "cohort": {
        "id_col": "PTID",
        "target_col": "TARGET",
        "story_date_col": "EXAMDATE_STORY",
        "ecog_date_col": "COLDATE",
        "max_ecog_lookback_days": 30,
        "ecog_before_or_on_story": True,
        "ecog_aggregation": "median",
        "ecog_scale": {
            "1": "Better or no change",
            "2": "Questionable or occasionally worse",
            "3": "Consistently a little worse",
            "4": "Consistently much worse",
            "9": "I don't know or not applicable (recoded to NaN)",
        },
        "ecog_missing_code": 9,
        "expected_primary_assessments": 77,
        "expected_primary_participants": 71,
        "expected_class_counts_participants": {"CN": 49, "MCI": 22},
    },
    "features": {
        "story_prefix": "STORY",
        "story_excluded_terms": ["PLAYBACK_COUNT", "GMATCH", "LANGUAGE"],
        "expected_story_feature_count": 51,
        "ecog_all_items": [f"ECOGPT{i}" for i in range(1, 13)],
        "ecog_reduced_items": [f"ECOGPT{i}" for i in range(3, 13)],
        "baseline_feature_sets": {
            "storyteller_only": {
                "use_storyteller": True,
                "ecog_items": [],
                "n_features": 51,
            },
            "ecog_pt_items_3_to_12_only": {
                "use_storyteller": False,
                "ecog_items": [f"ECOGPT{i}" for i in range(3, 13)],
                "n_features": 10,
            },
            "full_fusion_all_12_ecog": {
                "use_storyteller": True,
                "ecog_items": [f"ECOGPT{i}" for i in range(1, 13)],
                "n_features": 63,
            },
        },
    },
    "preprocessing": {
        "imputer": {
            "strategy": "median",
            "add_indicator": True,
        },
        "scaler": "StandardScaler",
    },
    "model": {
        "type": "LogisticRegression",
        "penalty": "l2",
        "C": 1.0,
        "max_iter": 5000,
        "class_weight": "balanced",
        "solver": "liblinear",
        "random_state": 42,
        "probability_threshold": 0.50,
    },
    "validation": {
        "cv_strategy": "StratifiedGroupKFold",
        "n_splits": 5,
        "shuffle": True,
        "primary_seed": 42,
        "seeds": [11, 21, 42, 52, 62, 72, 82, 92, 102, 112],
        "participant_aggregation": "mean_oof_probability",
    },
    "studies": {
        "study_1_historical_prior_ecog_baseline": {
            "notebook": "src/03_train_temporal_and_ablation.ipynb",
            "output_dir": "results/historical_prior_ecog_classification",
        },
        "study_2_temporal_window_sensitivity": {
            "notebook": "src/03_train_temporal_and_ablation.ipynb",
            "windows_days": {
                "30_days": 30,
                "90_days": 90,
                "180_days": 180,
                "365_days": 365,
                "historical": None,
            },
            "output_dir": "results/temporal_window_sensitivity",
        },
        "study_3_fixed_30day_cohort_comparison": {
            "notebook": "src/03_train_temporal_and_ablation.ipynb",
            "fixed_cohort_window_days": 30,
            "exports_primary_cohort": "data/processed/primary_multimodal_30day_cohort.csv",
            "comparison_windows_days": {
                "30_days": 30,
                "90_days": 90,
                "180_days": 180,
                "365_days": 365,
                "historical": None,
            },
            "output_dir": "results/common_cohort_temporal_comparison",
        },
        "all_pairs_sensitivity_extension": {
            "notebook": "src/06_dataset4_all_pairs_extension.ipynb",
            "dataset": "data/processed/dataset4_multimodal_cn_vs_noncn.csv",
            "target_definition": "CN (0) vs MCI/dementia (1)",
            "expected_pairs": 877,
            "expected_participants": 221,
            "expected_stable_label_participants": 218,
            "temporal_restriction": None,
            "output_root": "results/dataset4_setup",
        },
    },
}


PROPOSED_MODEL_CONFIG = {
    "experiment_group": "proposed_model",
    "model_name": "Reduced Multimodal Fusion (Storyteller + ECog-12 PT Self-Report items 3-12)",
    "project": {
        "name": "ADNI4 Speech and Participant Self-Reported Everyday Cognition (ECog-12 PT) MCI Detection",
        "ecog_source_table": "ECOG12PT (Everyday Cognition 12 Item - Participant Self-Report)",
        "primary_cohort_unit": "assessment",
        "evaluation_unit": "participant",
        "target_encoding": {"CN": 0, "MCI": 1},
    },
    "paths": {
        "raw_data_dir": "data/raw",
        "processed_data_dir": "data/processed",
        "notebooks_dir": "src",
        "results_root": "results",
        "figures_root": "figures",
        "datasets": {
            "storyteller_source": "data/processed/dataset1_storyteller.csv",
            "ecog_source": "data/processed/dataset2_ecog.csv",
            "primary_multimodal_30d": "data/processed/primary_multimodal_30day_cohort.csv",
        },
    },
    "cohort": {
        "id_col": "PTID",
        "target_col": "TARGET",
        "story_date_col": "EXAMDATE_STORY",
        "ecog_date_col": "COLDATE",
        "max_ecog_lookback_days": 30,
        "ecog_before_or_on_story": True,
        "ecog_aggregation": "median",
        "ecog_missing_code": 9,
        "expected_primary_assessments": 77,
        "expected_primary_participants": 71,
        "expected_class_counts_participants": {"CN": 49, "MCI": 22},
    },
    "features": {
        "story_prefix": "STORY",
        "story_excluded_terms": ["PLAYBACK_COUNT", "GMATCH", "LANGUAGE"],
        "n_story_features": 51,
        "ecog_included_items": [f"ECOGPT{i}" for i in range(3, 13)],
        "ecog_excluded_items": ["ECOGPT1", "ECOGPT2"],
        "ecog_key_item_definitions": {
            "ECOGPT1": "Remembering where you have placed personal items or objects (excluded)",
            "ECOGPT2": "Remembering the current date or day of the week (excluded)",
            "ECOGPT8": "Planning a trip, vacation, or outing (included; #1 SHAP)",
            "ECOGPT9": "Keeping living and work space (such as a home office, garage, or pantry) organized (included; #3 SHAP)",
        },
        "n_ecog_features": 10,
        "total_features": 61,
    },
    "preprocessing": {
        "imputer": {
            "strategy": "median",
            "add_indicator": True,
        },
        "scaler": "StandardScaler",
    },
    "model": {
        "type": "LogisticRegression",
        "penalty": "l2",
        "C": 1.0,
        "max_iter": 5000,
        "class_weight": "balanced",
        "solver": "liblinear",
        "random_state": 42,
        "probability_threshold": 0.50,
    },
    "validation": {
        "cv_strategy": "StratifiedGroupKFold",
        "outer_n_splits": 5,
        "inner_n_splits": 4,
        "shuffle": True,
        "primary_seed": 42,
        "seeds": [11, 21, 42, 52, 62, 72, 82, 92, 102, 112],
        "participant_aggregation": "mean_oof_probability",
    },
    "studies": {
        "study_4_ecog_item_ablation": {
            "notebook": "src/03_train_temporal_and_ablation.ipynb",
            "method": "leave_one_ecog_item_out",
            "output_dir": "results/ecog_item_ablation",
        },
        "study_5_ecog_ablation_stability": {
            "notebook": "src/03_train_temporal_and_ablation.ipynb",
            "seeds": [11, 21, 42, 52, 62, 72, 82, 92, 102, 112],
            "output_dir": "results/ecog_ablation_stability",
        },
        "study_6_nested_feature_selection": {
            "notebook": "src/03_train_temporal_and_ablation.ipynb",
            "outer_splits": 5,
            "inner_splits": 4,
            "selection_metric": "inner_participant_oof_roc_auc",
            "candidate_feature_sets": [
                "Full_fusion",
                "Without_ECOGPT1",
                "Without_ECOGPT2",
                "Without_ECOGPT1_ECOGPT2",
            ],
            "output_dir": "results/nested_feature_selection",
        },
        "study_7_primary_model_comparison_uncertainty": {
            "notebook": "src/04_evaluate_models_and_interpretation.ipynb",
            "frozen_models": [
                "Storyteller only",
                "Full fusion",
                "Fusion without ECOGPT1",
                "Fusion without ECOGPT1 and ECOGPT2",
                "Fusion without ECOGPT2",
            ],
            "uncertainty_method": "paired_participant_bootstrap_auc",
            "output_dir": "results/final_model_comparison_uncertainty/results",
        },
        "study_8_modality_and_feature_interpretation": {
            "notebook": "src/04_evaluate_models_and_interpretation.ipynb",
            "n_permutations": 20,
            "random_state": 2026,
            "output_dir": "results/feature_interpretation/results",
        },
        "study_9_feature_importance_verification": {
            "notebook": "src/04_evaluate_models_and_interpretation.ipynb",
            "output_dir": "results/feature_importance_verification",
        },
        "study_10_feature_correlation_redundancy": {
            "notebook": "src/04_evaluate_models_and_interpretation.ipynb",
            "output_dir": "results/feature_correlation_redundancy",
        },
        "study_11a_calibration_prauc_thresholds_and_dca": {
            "notebook": "src/04_evaluate_models_and_interpretation.ipynb",
            "metrics": ["ROC_AUC", "PR_AUC", "Brier_Score", "ECE_5bin", "Net_Benefit"],
            "operating_points": ["fixed_0.50", "youden_j", "high_sensitivity_ge_0.80"],
            "output_dir": "results/publication_calibration_and_dca",
            "figure": "figures/fig_calibration_and_decision_curve.png",
        },
        "study_11b_participant_demographics_table1": {
            "notebook": "src/04_evaluate_models_and_interpretation.ipynb",
            "source_raw_table": "data/raw/*PTDEMOG*.csv",
            "output_dir": "results/publication_demographics_table1",
        },
        "study_12_and_13_shap_explainability_and_directionality": {
            "notebook": "src/05_inference_shap_explainability.ipynb",
            "explainer": "shap.LinearExplainer",
            "compared_models": [
                "Full fusion",
                "Fusion without ECOGPT1 and ECOGPT2",
            ],
            "directionality_metric": "spearman_raw_feature_vs_shap",
            "output_dir": "results/primary_cohort_shap_explainability",
            "figure": "figures/fig_shap_top12_reduced_fusion.png",
        },
    },
}


def write_configs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "baseline.yaml": BASELINE_CONFIG,
        "proposed_model.yaml": PROPOSED_MODEL_CONFIG,
    }
    for filename, payload in files.items():
        target = output_dir / filename
        with open(target, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)
        print(f"Wrote {target}")


if __name__ == "__main__":
    write_configs(Path("configs"))