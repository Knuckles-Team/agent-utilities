# AU-RETRIEVAL — Tasks

- [ ] Publish a synthetic versioned corpus and manifest, with gold spans, splits and visibility labels.
- [ ] Reuse existing corpus, hierarchy, context and citation components behind one metric scorer.
- [ ] Add a disposable EG served hybrid/vector+graph baseline with exact actor/snapshot parity; keep offline lexical baselines honestly labelled.
- [ ] Implement citation and moved-section negative cases plus latency/update-cost collection.
- [ ] Predeclare held-out acceptance budgets; run paired comparisons and publish inconclusive results as such.
- [ ] Run focused/full tests and CCCC, jscpd, Dupehound, KISS review; add exact-revision receipts before status change.
- [ ] Gate retrieval proposals from the study pipeline on a paired `AU-RETRIEVAL-R001` result in `tests/retrieval/test_study_admission.py`. Closes AU-RETRIEVAL-R002.
- [ ] Add iterative memory expansion with a declared budget, typed abstention and trace evidence. Prove it through the served memory retrieval operation. Closes AU-RETRIEVAL-R003.
- [ ] Make the executable retrieval loop request rounds on an engine `insufficient` verdict and abstain under the received strict policy. Prove it through the served `executable_rag` action. Closes AU-RETRIEVAL-R004.
- [ ] Make both iterative loops stop on the engine stop verdict and trace its reason. Closes AU-RETRIEVAL-R005.
- [ ] Apply the engine arbitration verdict at the answer step, abstaining with both candidates. Closes AU-RETRIEVAL-R006.
