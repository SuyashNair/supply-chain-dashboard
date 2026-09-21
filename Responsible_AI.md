# Responsible AI Checklist & Report

## 1. Fairness & Bias
- **Assessment:** The supply chain model relies on regional data (`country`, `state`). There is a risk that the model might unfairly penalize shipments to certain regions (e.g., rural areas or developing nations) by systematically predicting higher delay risks, potentially leading to deprioritization of those routes.
- **Mitigation:** We monitor accuracy across different countries and regions to ensure false positive rates (predicting delays when there are none) are consistent across all demographic/geographic segments.

## 2. Privacy & Data Security
- **Assessment:** This dataset contains aggregate demand and retail sales data. While it does not contain Personally Identifiable Information (PII) like customer names or addresses, geographic aggregation could inadvertently reveal supplier business strategies.
- **Mitigation:** Strict access controls on the dataset. Ensure no PII is fed into the API endpoint during production use.

## 3. Explainability & Transparency
- **Assessment:** Stakeholders (warehouse managers, logistics coordinators) need to trust the model. A black-box Random Forest is hard to interpret.
- **Mitigation:** We have integrated SHAP (SHapley Additive exPlanations) into our Streamlit dashboard to show exactly which features (e.g., `quantity`, `month`) drive the delay prediction for every single order.

## 4. Consent & Compliance
- **Assessment:** The dataset is sourced openly for research/experimental purposes.
- **Mitigation:** If deployed in production, data collection must comply with GDPR/CCPA regulations, ensuring supply chain partners consent to their delivery metrics being used for predictive analytics.
