import shap


def get_shap_values(model_pipeline, X_train, X_test_instance):
    preprocessor = model_pipeline.named_steps['preprocessor']
    classifier   = model_pipeline.named_steps['classifier']

    X_train_t = preprocessor.transform(X_train)
    X_test_t  = preprocessor.transform(X_test_instance)

    explainer   = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_test_t)
    return explainer, shap_values, X_train_t, X_test_t


def get_human_readable_explanation(shap_values, feature_names, instance_idx=0):
    instance_shap = shap_values[1][instance_idx]
    impacts = sorted(zip(feature_names, instance_shap),
                     key=lambda x: abs(x[1]), reverse=True)

    out = []
    for feature, impact in impacts[:3]:
        clean = (feature.replace('num__', '')
                        .replace('cat__', '')
                        .replace('_', ' ').title())
        verb = "increased" if impact > 0 else "decreased"
        out.append(f"**{clean}** {verb} the risk.")
    return out
