# AU-RETRIEVAL — Tasks

- [ ] Publish a synthetic versioned corpus and manifest, with gold spans, splits and visibility labels.
- [ ] Reuse existing corpus, hierarchy, context and citation components behind one metric scorer.
- [ ] Add a disposable EG served hybrid/vector+graph baseline with exact actor/snapshot parity; keep offline lexical baselines honestly labelled.
- [ ] Implement citation and moved-section negative cases plus latency/update-cost collection.
- [ ] Predeclare held-out acceptance budgets; run paired comparisons and publish inconclusive results as such.
- [ ] Run focused/full tests and CCCC, jscpd, Dupehound, KISS review; add exact-revision receipts before status change.
- [ ] Gate retrieval proposals from the study pipeline on a paired `AU-RETRIEVAL-R001` result in `tests/retrieval/test_study_admission.py`. Closes AU-RETRIEVAL-R002.
- [ ] Add iterative memory expansion with a declared budget, typed abstention and trace evidence. Prove it through the served memory retrieval operation. Closes AU-RETRIEVAL-R003.
- [ ] Add typed evidence sufficiency, a strict fallback policy and abstention to the executable retrieval program; prove it through the served `executable_rag` action. Closes AU-RETRIEVAL-R004.
- [ ] Add a calibrated confidence margin to the iterative stopper and pass the round confidence from both production callers. Closes AU-RETRIEVAL-R005.
- [ ] Add prior-versus-evidence arbitration with margin abstention at the answer step. Closes AU-RETRIEVAL-R006.
