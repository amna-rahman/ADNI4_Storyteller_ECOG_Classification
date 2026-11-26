# AI Use Disclosure

## Statement of Responsibility and Authorship
Generative AI tools were used solely as an auxiliary assistance resource during the development of this repository and accompanying manuscript. All research design decisions, analytical choices, model specifications, data handling, interpretations, and final written content were directed, executed, reviewed, and verified by the author.

No AI system qualifies for authorship, co-authorship, or intellectual credit for this work. The author assumes full and sole responsibility for the scientific integrity, accuracy, reproducibility, and originality of all code, figures, tables, and text in this project.

## Scope of AI Assistance
AI assistance was limited to supportive technical and editorial tasks, including:

- **Code organization and debugging:** Assisting with modularizing existing Jupyter notebooks into the `src/` pipeline structure, resolving import/path issues, and formatting configuration files (`configs/*.yaml`).
- **Visualization and formatting:** Assisting with `matplotlib` plotting code for black-and-white publication figures and formatting Markdown documentation (`README.md`, `data/README.md`) and manuscript tables.
- **Editorial refinement:** Assisting with grammar, clarity, conciseness, and structural alignment with journal formatting guidelines.

## Human Verification and Data Governance
- **Independent execution and verification:** Every notebook (`01_preprocessing.ipynb` through `06_dataset4_all_pairs_extension.ipynb`), statistical test, cross-validation run, and SHAP analysis was executed locally by the author, and all reported metrics were verified directly against locally generated output files.
- **Data dictionary and clinical checks:** Variable definitions, exclusion rules, and ADNI4 table semantics (including verification of `ECOG12PT` as Participant Self-Report) were cross-checked against the official ADNI4 data dictionary (`DATADIC`) and ADNI documentation.
- **ADNI data compliance:** All raw ADNI4 clinical, questionnaire, and Storyteller datasets remain strictly governed by the ADNI Data Use Agreement and are excluded from version control via `.gitignore`.
