# Project: Pneumonia Segmentation Chapter 3 Thesis Writing

## Architecture
This project consists of writing and compiling Chapter 3 (Research Methodology) of a undergraduate thesis in LaTeX, along with its supporting UML/Flowchart diagrams and bibliography.

Key components:
- **`doc/section/bab_3.tex`**: The standalone LaTeX chapter for Bab 3.
- **`doc/skripsi.tex`**: The main thesis document including Bab 3 inline.
- **`doc/references.bib`**: BibTeX bibliography containing reference definitions.
- **`doc/plantuml/`**: Directory where PlantUML diagram sources (.puml) are saved.
- **`doc/journal/markdown/`**: Source academic literature for methodology extraction and bibliography citation.

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|---|---|---|---|
| 1 | Literature Review & Extraction | Analyze literature in `doc/journal/markdown/` to extract methodologies for U-Net, EfficientNet-B4, Grad-CAM, PSPNet lung masking, and Gradio UI. (Conv ID: 95176d91-333a-4660-ba38-273a218a0c63) | None | DONE |
| 2 | Visualizations (PlantUML) | Create at least 2 PlantUML diagrams (flowchart and system architecture) in `doc/plantuml/` with placeholders in LaTeX. (Conv ID: f99e73b5-4cd8-42fd-aadf-b042340df958) | None | DONE |
| 3 | LaTeX Writing & Drafting | Write the comprehensive LaTeX text for Bab 3 in `doc/section/bab_3.tex` and `doc/skripsi.tex`. (Conv ID: 7347dc3a-8c6b-4561-809d-9b79ef31c246) | M1 | DONE |
| 4 | Bibliography Update | Update `doc/references.bib` with citations used in Bab 3. (Conv ID: 7347dc3a-8c6b-4561-809d-9b79ef31c246) | M3 | DONE |
| 5 | Compilation & Verification | Run LaTeX builds via Makefiles to verify compilation and syntax. (Conv ID: 46bc97a1-138b-424f-9ef5-6aef90418ad0) | M3, M4 | IN_PROGRESS |

## Code Layout
- `doc/`
  - `skripsi.tex`: Main document
  - `references.bib`: BibTeX bibliography file
  - `section/`
    - `bab_3.tex`: Chapter 3 document
  - `plantuml/`
    - `flowchart_penelitian.puml`: Methodology flowchart
    - `arsitektur_sistem.puml`: Segmentation pipeline / network architecture
  - `journal/markdown/`: Folder of reference markdown papers
