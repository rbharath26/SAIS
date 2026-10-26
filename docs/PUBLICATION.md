# Publication checklist

This repository contains local draft materials only. Nothing has been submitted
to arXiv, ResearchGate, GitHub, or any other public service.

## Before release

1. Confirm author name, affiliation, contact address, and contribution history.
2. Complete empirical evaluation using a licensed human-scored dataset. Do not
   claim unmeasured correlation results.
3. Confirm that `LICENSE`, `EULA.md`, package metadata, and commercial terms
   have been reviewed by counsel.
4. Build the LaTeX document:

   ```bash
   cd docs
   pdflatex SAIS_preprint.tex
   bibtex SAIS_preprint
   pdflatex SAIS_preprint.tex
   pdflatex SAIS_preprint.tex
   ```

5. Review the source, PDF, citation record, and benchmark result artifacts.
   Do not publish proprietary source or commercial build artifacts to a public
   GitHub repository.

## Submission options

- **arXiv:** The author must sign in, meet the relevant subject-area submission
  requirements (which can include endorsement), select a license, and upload
  the final source/PDF. Verify current arXiv requirements before submitting.
- **ResearchGate:** Upload only a version the author has the right to share and
  accurately identify it as a preprint when it is not peer reviewed.
- **Private repository:** Keep implementation, release artifacts, license keys,
  customer information, and commercial documentation private. A public paper
  does not require public source release.

External submission is an author action. Verify the final metadata and
submission terms before publishing.

## Current local validation status

The source package, unit tests, wheel build, and benchmark command parsing
have been validated. No SAIS correlation figures are included in the preprint.
This draft must not claim empirical benchmark results until a semantic
embedding model has been successfully obtained and the benchmark reports have
been generated.
