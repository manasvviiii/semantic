\# Semantic Model Evaluation â€” V5



\## 1. Purpose



V5 is the semantic classification component of the Vak-Siddhi audio analysis pipeline. It classifies transcribed messages into:



\- `legitimate` â€” label 0

\- `scam\_coercion` â€” label 1



The audio pipeline uses Whisper for transcription and V5 for semantic classification.



\## 2. Model Architecture



\- Text encoder: `sentence-transformers/all-MiniLM-L6-v2`

\- Embeddings: attention-mask mean pooling followed by L2 normalization

\- Classifier: Logistic Regression

\- Inference device: CPU

\- Maximum token length: 256

\- Saved artifact: `models/semantic\_slm\_security\_v5.joblib`



\## 3. Verification Results



\- Existing classifier tests: 3 passed

\- V5 inference tests: 5 passed

\- Combined test suite: 8 passed, 0 failed

\- Audio smoke test: transcription and V5 classification completed successfully for `data/sample.wav`



The smoke test classified the sample appointment-confirmation recording as `legitimate`, with a model probability of approximately 95.46% for that predicted class.



Four additional handcrafted smoke-test examples produced their expected classifications. These checks are basic functional tests, not a representative performance benchmark.



\## 4. Known Limitations



\- Refund-related legitimate messages can be incorrectly classified as coercive.

\- Some coercive delivery messages receive weaker model scores than coercive refund messages.

\- Evaluation examples are limited and include synthetic data.

\- Model probabilities are not guaranteed to be calibrated measures of real-world fraud risk.

\- A prediction is not proof of fraud and must not be used as a definitive accusation.



\## 5. Preservation and Reproducibility



\- V5 is used by `semantic/slm\_inference.py`.

\- `semantic/pipeline.py` imports V5 inference for audio classification.

\- The existing TF-IDF implementation in `semantic/classifier.py` remains separate.

\- V4, V5 and V6 model artifacts should not be overwritten during integration checks.



\## 6. Next Steps



1\. Run the complete unit test suite after code changes.

2\. Test with additional representative, appropriately sourced audio.

3\. Record false positives and false negatives honestly.

4\. Set up Git and commit the intended source files and documentation.

5\. Do not claim production readiness without broader independent evaluation.
