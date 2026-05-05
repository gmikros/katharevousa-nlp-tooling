# Building a Reproducible Katharevousa Greek NLP Pipeline:
# From OCR Reconstruction and LLM-Assisted Gold Data to Dependency Parsing

**Version:** v1 draft  
**Target venues:** arXiv preprint (first), then journal/conference adaptation  
**Authors:** George Mikros, Fotios Fitsilis, et al. (confirm final order/affiliations)

---

## Abstract

Katharevousa Greek remains under-served by modern Natural Language Processing (NLP) tools, despite its central role in archival, legal, and parliamentary corpora. We present a reproducible end-to-end pipeline for building a dependency parsing benchmark and model stack for Katharevousa from noisy historical sources. Starting from OCR-derived parliamentary material, we apply targeted text reconstruction, large-scale LLM-assisted annotation, and quality-controlled CoNLL-U consolidation to produce a finalized gold dataset of 1,697 sentences. We evaluate multiple modeling paths: a feature-based logistic baseline, multilingual transformer parsers, mBERT variants, and Stanza baselines (pretrained and custom-trained). The best overall model is an XLM-R-based parser (`v1_opt`) with UPOS 0.8893, DEPREL F1 0.7250, UAS 0.6098, and LAS 0.5162 on a fixed held-out split. Custom Stanza training substantially improves over pretrained Stanza on LAS but remains below the transformer best. Beyond model performance, the study contributes a practical methodology for low-resource historical language NLP: robust batch annotation operations, resume-safe data generation, explicit split manifests, and comparative error-driven experimentation. We release artifacts and scripts to support reproducibility and further research.

---

## 1. Introduction

Historical and formal Greek registers, especially Katharevousa, pose a structural mismatch for modern Greek NLP systems trained primarily on Demotic corpora. This mismatch affects tokenization, morphology, and especially syntactic parsing in long institutional prose. The problem is not only technical: it directly impacts computational humanities and political-text scholarship using transitional-era Greek archives.

This paper documents the design and evaluation of a dedicated Katharevousa NLP workflow, built as a reproducible research pipeline rather than an isolated model run. We focus on:

1. High-quality dataset construction under annotation constraints.
2. Controlled model comparison under fixed train/test splits.
3. Transparent reporting of successes, regressions, and engineering trade-offs.

Our primary objective is to establish a release-ready baseline suitable for open dissemination (Hugging Face) while preserving scientific auditability.

---

## 2. Background and Motivation

Your earlier DSH manuscript establishes the corpus context and demonstrates that post-junta parliamentary questions are linguistically rich but computationally difficult due to OCR noise, historical register, and morphology-syntax divergence from Demotic Greek. It also shows that hybrid spaCy+Stanza processing is analytically useful but not sufficient as a dedicated syntactic parser for Katharevousa.

This study extends that line of work by moving from feature extraction for discourse analysis to supervised dependency parsing with explicit model-selection criteria.

---

## 3. Data and Corpus Construction

### 3.1 Source material

The upstream corpus context follows the existing parliamentary archive pipeline described in the prior DSH draft: OCR-derived written parliamentary questions from the early post-junta period (1976-1977), in formal Katharevousa.

### 3.2 Reconstruction and preprocessing

Before syntactic annotation, we implemented reconstruction-aware preprocessing to address:

- OCR artifacts.
- line-break hyphenation and split-word restoration.
- sentence segmentation failure in long enumerative parliamentary text.
- malformed JSON and inconsistent model outputs in LLM-based annotation loops.

### 3.3 LLM-assisted gold creation

We adopted a GPT-centered annotation pipeline with:

- strict schema-constrained outputs (UD-style token annotations).
- validation and fallback parsing.
- resume-safe batch orchestration and targeted retry passes.
- cost/rate monitoring and interruption robustness.

### 3.4 Final frozen gold snapshot

A deterministic freeze step merged primary batches and retries into a deduplicated final CoNLL-U snapshot:

- **Final gold sentences:** 1,697  
- **Source breakdown:** 1,565 first-pass batch + 132 retry replacements  
- **Frozen file:** `data/processed/final_gold/gold_final.conllu`  
- **Manifest:** `data/processed/final_gold/snapshot_manifest.json`

---

## 4. Annotation and Representation

We follow Universal Dependencies style annotation (token-level POS, FEATS, HEAD, DEPREL) in CoNLL-U, with project-specific handling for Katharevousa edge cases (archaizing morphology and long-distance dependencies). The study preserves fixed split manifests for reproducibility and exact reruns.

---

## 5. Experimental Design

### 5.1 Split protocol

- Fixed sentence split with seed 42.
- Train: 1,357 sentences.
- Test: 340 sentences.
- Test tokens: 4,093.
- Split manifest stored and reused across all comparisons.

### 5.2 Metrics

- UPOS accuracy
- DEPREL weighted F1
- UAS
- LAS

### 5.3 Model families evaluated

1. **Feature-based logistic parser** (improved head-label learning, hard-case experiments).
2. **Transformer parser (XLM-R)** with joint UPOS/arc/deprel heads.
3. **mBERT transformer variants** under matched split.
4. **Stanza pretrained Greek baseline**.
5. **Custom-trained Stanza POS+depparse** on project gold data.

---

## 6. Results

### 6.1 Main comparative table

| Model | UPOS | DEPREL F1 | UAS | LAS |
|---|---:|---:|---:|---:|
| Logistic baseline (`final_gold_eval_report_v3`) | 0.9040 | 0.7451 | 0.5781 | 0.5072 |
| Transformer XLM-R (`transformer_parser_v1_opt`) | 0.8893 | 0.7250 | **0.6098** | **0.5162** |
| mBERT best (`transformer_parser_mbert_v2`) | 0.8260 | 0.6076 | 0.5886 | 0.4537 |
| Stanza pretrained baseline (`stanza_baseline_report`) | 0.6125 | 0.4242 | 0.6079 | 0.3396 |
| Stanza custom (`stanza_custom_v1_report`, 600 steps) | 0.7694 | 0.6588 | 0.5756 | 0.4943 |

### 6.2 Interpretation

- `v1_opt` is the strongest **overall parser** in structural terms (best UAS/LAS among transformer-family and framework comparisons).
- The logistic model remains very competitive in UPOS/DEPREL and near-best LAS, showing that lexical-feature engineering still has value at this dataset scale.
- mBERT underperforms XLM-R across all metrics in these experiments.
- Pretrained Stanza is surprisingly close in UAS but weak in LAS/DEPREL; custom training narrows the gap substantially but does not surpass `v1_opt`.

---

## 7. Error Analysis and Ablation Journey

The experimentation path surfaced several robust lessons:

- **Scheduler and weighting sensitivity:** small tuning changes often degraded all four metrics.
- **Hard-case reweighting:** did not consistently improve generalization in this dataset size regime.
- **Tokenizer truncation alignment:** strict old/new token index mapping is mandatory for stable dependency training.
- **Framework constraints:** Stanza training is sensitive to CoNLL-U structural validity (cycles, format strictness), adding engineering overhead.

These negative and mixed findings are scientifically useful and should be documented explicitly rather than omitted.

---

## 8. Scientific Contribution

This work contributes:

1. A practical recipe for building a high-quality syntactic gold set for historical Greek with constrained manual effort.
2. A reproducible benchmark split and reporting framework for Katharevousa dependency parsing.
3. Cross-framework evidence that modern multilingual transformers (XLM-R) currently provide the best parsing trade-off in this setting.
4. A bridge from digital-humanities corpus analysis to release-ready NLP infrastructure.

---

## 9. Limitations

- Gold data is LLM-assisted and not fully manually adjudicated.
- Domain is parliamentary-formal Katharevousa; transfer to other genres is untested.
- Current data volume (1,697 sentences) is strong for pilot benchmarking but still modest for large parser generalization.
- UD consistency noise from OCR/reconstruction propagates into training uncertainty.

---

## 10. Ethics and Responsible Use

- Historical-political texts can encode ideological and institutional bias.
- Model outputs should not be used as sole evidence in legal/historical interpretation.
- Releases should include uncertainty framing and intended-use constraints.

---

## 11. Reproducibility and Release Assets

Recommended release package:

- Frozen gold dataset snapshot and manifest.
- Split manifest used for all reported comparisons.
- Training/eval scripts and report JSON files.
- Model card with explicit benchmark table and limitations.

This supports transparent replication and faster community iteration.

---

## 12. Conclusion

A reproducible Katharevousa parsing pipeline is feasible with constrained annotation budgets when LLM-assisted data generation is paired with strict validation and robust operations. Among tested systems, `v1_opt` is the best release candidate for first open publication, while custom Stanza is a useful secondary baseline. The project is now at a strong transition point from experimental prototyping to public research artifact.

---

## Appendix A - Figure Plan for Overleaf (final paper)

1. **Pipeline overview flowchart**  
   OCR -> reconstruction -> LLM annotation -> freeze -> train/eval -> model selection.

2. **Dataset funnel chart**  
   Raw OCR -> validated corpus -> frozen gold -> train/dev/test split.

3. **Metric comparison grouped bar chart**  
   Models on UPOS/DEPREL/UAS/LAS (single panel or two panels: morph vs syntax).

4. **Training curves**  
   Transformer `v1_opt` loss across epochs; optional custom Stanza dev trend.

5. **Error profile plot**  
   LAS/UAS by sentence length and head distance bucket (from parser error reports).

6. **Confusion heatmap (DEPREL)**  
   Top deprel confusions for best transformer and best non-transformer baseline.

---

## Appendix B - Table Plan for Overleaf (final paper)

1. **Corpus and split statistics**
2. **Model configuration summary**
3. **Main benchmark table** (included above)
4. **Ablation and tuning outcomes** (small controlled grid, v2/v2b, mBERT variants)
5. **Framework-level comparison** (XLM-R vs mBERT vs Stanza pretrained/custom)
6. **Failure/repair log table** (key engineering issues and fixes)

---

## Appendix C - Suggested Overleaf snippets (starter)

```latex
\begin{table}[t]
\centering
\caption{Main benchmark results on fixed test split (n=340 sentences).}
\begin{tabular}{lcccc}
\toprule
Model & UPOS & DEPREL F1 & UAS & LAS \\
\midrule
Logistic baseline (v3) & 0.9040 & 0.7451 & 0.5781 & 0.5072 \\
XLM-R (v1\_opt) & 0.8893 & 0.7250 & \textbf{0.6098} & \textbf{0.5162} \\
mBERT (best) & 0.8260 & 0.6076 & 0.5886 & 0.4537 \\
Stanza pretrained & 0.6125 & 0.4242 & 0.6079 & 0.3396 \\
Stanza custom (600 steps) & 0.7694 & 0.6588 & 0.5756 & 0.4943 \\
\bottomrule
\end{tabular}
\end{table}
```

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/model_metric_comparison.pdf}
\caption{Comparison of UPOS, DEPREL F1, UAS, and LAS across model families.}
\end{figure}
```

---

## Notes for next draft (v2)

- Replace placeholder citation keys with full bibliography entries.
- Add significance testing across model families if you want inferential claims (e.g., bootstrap confidence intervals for LAS/UAS).
- Expand qualitative error examples (2-3 representative sentences) for interpretability.
- Align wording with target venue guidelines (DSH vs computational linguistics venue).
