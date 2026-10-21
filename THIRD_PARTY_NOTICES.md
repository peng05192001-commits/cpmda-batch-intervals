# Attribution and scope

The statistical reference is **Conformal Prediction with Missing Values**, Margaux Zaffran, Aymeric Dieuleveut, Julie Josse and Yaniv Romano, ICML 2023, PMLR 202:40578-40604. https://proceedings.mlr.press/v202/zaffran23a.html

Reference implementation: https://github.com/mzaffran/ConformalPredictionMissingValues , commit e1ff0c83a4943e6468b78bfec49af64fc61c3561, prediction.py SHA256 996e9c38779df328fc0a42b94641d14dccb1f02505b7ad9e342f0463ace90abf . The upstream repository uses the MIT License; its notice is retained in UPSTREAM_LICENSE_MIT.txt. The full upstream implementation and its unrelated dependencies are not vendored here. This repository implements the paper's fixed regression score/endpoint computation and explicit comparison kernels; it does not claim a new general sorted-list selection principle or a new coverage theorem.

NumPy, Numba, scikit-learn and joblib are installed as dependencies, not redistributed source trees. They retain their respective licenses.

The public model files are minimal exports of the author's frozen fits, retaining only prep and models. Unused historical model branches were removed; no model was refitted. model_export_checks.json records the original/public hashes and exact prediction comparison. Model artifacts are supplied for this computational benchmark, not a claim of forecast validity.

Data licensing is separate: see DATA_LICENSE.md. Recorded experiment results are unchanged apart from replacing local absolute filesystem paths with archived-local-source/basename. These sanitized records therefore do not retain their original file hashes; source-code/input/model hash values inside the records are kept for provenance. Public executable files have their own manifest.
