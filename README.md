# **Design Specification: Multi‑Layer AI‑Generated Music Detection System**

## **1\. Project Overview**

Recent advances in generative audio models—such as Suno, Udio, MusicGen, and diffusion‑based systems—have made AI‑generated music increasingly realistic and difficult to distinguish from human‑composed works. Traditional detectors often rely on a single feature domain (e.g., spectral fingerprints, codec artifacts, or short‑window CNN classification), which makes them brittle, easy to evade, and poorly generalizable to unseen generators.

This project proposes a **multi‑layer, multi‑modal detection system** that integrates:

* **Spectral artifact analysis** (checkerboard patterns, periodic peaks, codec fingerprints)  
* **Temporal coherence modeling** (melodic continuity, structural segmentation, long‑range dependencies)  
* **Timbral and production‑level cues** (mixing/mastering inconsistencies, unnatural transients, FX artifacts)  
* **Micro‑signal anomaly detection** (upsampling artifacts, periodicity from transposed convolutions)  
* **Section‑level attribution** (intro, verse, chorus, bridge, instrumental break)

The system will operate at two levels:

### **Track‑level detection**

A global classifier determines whether the entire track is AI‑generated.

### **Section‑level attribution**

A segmentation‑aware model identifies *which parts* of the track exhibit the strongest AI signatures, enabling:

* Explainability  
* Partial‑fake detection  
* Forensic analysis  
* Model‑agnostic generalization

This design draws inspiration from recent breakthroughs in AI‑music detection, Fourier‑domain artifact analysis, segment‑transformer architectures, and multi‑generator datasets.

---

## **2\. System Architecture (High‑Level)**

### **2.1 Multi‑Layer Detection Pipeline**

1. **Preprocessing & Segmentation**

   * Beat tracking, downbeat detection  
   * 4‑bar segmentation (following Segment Transformer literature)  
   * Multi‑resolution STFTs (short/medium/long windows)  
2. **Feature Extraction**

   * **Spectral fingerprints** (Fourier peaks, periodic summation artifacts)  
   * **Melody/harmony embeddings** (MERT, Music2Vec, wav2vec‑style SSL models)  
   * **Timbral embeddings** (FXEncoder, spectral centroid/bandwidth, HNR)  
   * **Structural similarity matrices** (self‑similarity matrices, recurrence plots)  
3. **Modeling**

   * **Segment‑level classifier** (AudioCAT‑style cross‑attention encoder)  
   * **Full‑track structural model** (Segment Transformer or Fusion Segment Transformer)  
   * **Ensemble fusion** (late fusion of spectral, temporal, and timbral predictions)  
4. **Section‑Level Attribution**

   * Attention heatmaps  
   * Local anomaly scores  
   * Structural discontinuity detection  
   * Fourier‑domain artifact localization  
5. **Output**

   * Binary classification (AI vs. human)  
   * Section‑level AI probability timeline  
   * Explanation report (spectral peaks, structural anomalies, timbral inconsistencies)

---

## **3\. Project Timeline**

### **Week 1–2: Literature Review & Dataset Assembly**

* Review 15–20 papers (provided in Related Work)  
* Collect datasets: M6, SONICS, FakeMusicCaps, FMA, GTZAN, COSIAN  
* Build pipeline for downloading/processing Suno/Udio samples (if allowed)

### **Week 3–4: Preprocessing & Feature Extraction**

* Implement beat/downbeat tracking  
* Implement multi‑resolution STFT extraction  
* Extract SSL embeddings (MERT, Music2Vec, wav2vec2)  
* Extract Fourier‑domain artifact fingerprints

### **Week 5–6: Segment‑Level Classifier**

* Implement AudioCAT‑style cross‑attention model  
* Train on FakeMusicCaps \+ SONICS short‑segment data  
* Evaluate robustness to pitch‑shift, time‑stretch, codec compression

### **Week 7–8: Full‑Track Structural Model**

* Implement Segment Transformer or Fusion Segment Transformer  
* Train on SONICS full‑track data  
* Integrate structural similarity matrices

### **Week 9–10: Section‑Level Attribution**

* Implement attention‑based localization  
* Implement Fourier‑peak localization  
* Build visualization tools (heatmaps, timelines)

### **Week 11–12: Fusion & Final System**

* Combine spectral, temporal, and timbral models  
* Evaluate on M6 out‑of‑distribution generators  
* Write final report \+ prepare demo

---

## **4\. Tools & Technologies**

### **Programming & Frameworks**

* Python  
* PyTorch  
* Librosa/Essentia  
* Scikit-learn  
* Jupyter Notebooks

### **Feature Representations**

* CLAP (Contrastive Language–Audio Pretraining) embeddings as the primary representation, following Cros Vila et al. (2025)  
* Essentia low-level and mid-level descriptors (spectral centroid, loudness, BPM, harmonic features) for complementary analysis and ablation studies

### **Models and Classifiers**

* Classical ML classifiers operating on embeddings:  
  * Support Vector Machines (SVM)  
  * Random Forests  
  * k-Nearest Neighbors (k-NN)  
* Segment Transformer（Kim et al.）  
* Fusion Segment Transformer（Kim et al. 2026）  
* CNN / RNN / CRNN Models  
* SHAP, LIME, Grad‑CAM on spectrograms (explain models)

### **Audio Transformations & Robustness Testing**

* Resampling (e.g., 48 kHz → 22.05 kHz)  
* Low-pass and high-pass filtering  
* Pitch-shifting and time-stretching

### **Datasets**

* **M6** (multi‑generator, multi‑genre, multi‑lingual)  
* **SONICS** (Suno/Udio full‑track dataset)  
* **FakeMusicCaps** (short‑segment TTM dataset)  
* **FMA** (real music baseline)  
* **GTZAN** (genre diversity)  
* **COSIAN** (Japanese vocal music)  
* **Musical Instrument Sound Dataset** (instrument‑level analysis)  
* **Million Song Dataset (MSD)** or FMA as non-AI baselines

### **Visualization & Analysis**

* UMAP for embedding visualization  
* Confusion matrices, precision/recall/F1  
* Feature importance analysis for classical models

---

## **5\. Related Work**

Research on AI‑generated music detection has accelerated rapidly in response to the emergence of high‑fidelity generative audio models such as Suno, Udio, MusicGen, and diffusion‑based systems. Early work in this area focused primarily on **audio deepfake detection** and **speech spoofing** (e.g., Wu et al., 2017; Zang et al., 2024), but recent studies have begun to address the unique challenges posed by **full‑length musical content**, which exhibits richer structure, broader timbral variation, and more complex production pipelines than speech.

## **5.1.  Artifact‑Based Detection and Neural Codec Fingerprints**

A major line of work investigates spectral artifacts introduced by neural audio decoders. Afchar et al. (2024, 2025\) demonstrate that modern music generators—especially those using transposed convolutions or neural codecs such as Encodec and DAC—produce periodic spectral peaks, checkerboard patterns, and upsampling artifacts that can be exploited for detection. Their Fourier‑domain analysis shows that these artifacts arise from architectural constraints rather than training data, making them stable and model‑specific fingerprints. Simple linear models trained on averaged spectral fingerprints can achieve \>99% accuracy on known generators, though generalization to unseen models remains limited.

These findings align with earlier observations in image and speech deepfake detection, where convolutional upsampling leaves detectable periodicities. However, Afchar et al. also highlight that such detectors are fragile: minor manipulations such as pitch‑shifting, re‑encoding, or adding noise can collapse accuracy to near zero. This motivates detection systems that go beyond low‑level spectral cues.

## **5.2. Structural and Temporal Modeling of Music**

Music differs from speech in its long‑range temporal dependencies, hierarchical structure, and repeated motifs. Several works propose modeling these properties for AI‑music detection.

Rahman et al. (2025) introduce SpecTTTra, a spectro‑temporal transformer that tokenizes both time and frequency axes to capture long‑range dependencies. Their SONICS dataset—containing tens of thousands of Suno/Udio tracks—demonstrates that long‑context models outperform short‑window CNNs, especially for full‑length songs. However, SpecTTTra still struggles with unseen generators and is sensitive to distribution shifts.

Kim & Go (2025, 2026\) propose the Segment Transformer, which segments music into beat‑aligned units (e.g., 4‑bar chunks) and models inter‑segment relationships using self‑similarity matrices. This approach explicitly incorporates musical form, enabling detection of structural inconsistencies typical of AI‑generated music (e.g., unnatural repetition, lack of developmental arcs). Their two‑stage system—segment‑level feature extraction followed by structural modeling—achieves state‑of‑the‑art results on SONICS and AIME datasets.

These works collectively show that temporal coherence and structural segmentation are crucial for robust detection, especially as spectral artifacts become less reliable.

## **5.3. Timbral, Mixing, and Production‑Level Cues**

Beyond spectral fingerprints, several studies explore production‑level anomalies in AI‑generated music. FXEncoder‑based systems (Kim & Go, 2025\) analyze mixing/mastering characteristics such as dynamic range, transient behavior, and effect‑chain consistency, which differ between human‑engineered tracks and AI‑generated ones. Similarly, Li et al. (2024, 2025\) show that MGM (machine‑generated music) often exhibits unnatural harmonic‑to‑noise ratios, spectral bandwidth distributions, and instrumental balance, which can be used as discriminative features.

These approaches highlight that AI models often fail to replicate the nuanced, nonlinear workflows of human audio engineers—an underexplored but promising detection dimension.

## **5.4. Micro‑Signal and Low‑Level Anomaly Detection**

Several papers emphasize micro‑signal irregularities introduced by generative models:

* periodic aliasing from transposed convolutions (Afchar et al., 2025\)  
* unnatural zero‑crossing patterns (Li et al., 2024\)  
* inconsistencies in onset autocorrelation and transient sharpness (Cros Vila & Sturm, 2025\)  
* codec‑specific reconstruction artifacts (Afchar et al., 2024\)

These micro‑artifacts are often invisible to human listeners but detectable through high‑resolution spectral or temporal analysis. However, they are also the easiest for future models to eliminate, making them insufficient as a standalone detection strategy.

## **5.5. Lyrics‑Based and Multimodal Detection**

Frohmann et al. (2025) propose a complementary approach: detecting AI‑generated songs via lyrics transcripts. Using Whisper for transcription and LLM‑based text encoders (e.g., LLM2Vec), they show that lyric‑style features generalize better across audio manipulations and unseen generators than audio‑only detectors. This suggests that multimodal detection—combining audio, lyrics, metadata, and structural cues—may be necessary for long‑term robustness.

## **5.6. Symbolic and MIDI‑Based Detection**

Several works focus on symbolic music:

* Tantra & Wicaksana (2025) use LSTM/CNN models to detect AI‑generated classical MIDI compositions.  
* Dervakos et al. (2021) propose music‑theoretic heuristics (tonal networks, harmonic span, structural coherence) to evaluate generative models.  
* Cros Vila & Sturm (2023–2025) develop statistical and geometric representations (e.g., cosine contours, corpus‑level distance metrics) to compare symbolic melodies.

These studies show that symbolic representations reveal melodic continuity, harmonic logic, and structural development—areas where AI systems still struggle.

## **5.7. Dataset Development and Benchmarking**

Multiple papers contribute datasets and benchmarks:

* **SONICS** (Rahman et al., 2025): large‑scale Suno/Udio dataset with structured prompts.  
* **M6** (Li et al., 2024): multi‑generator, multi‑genre, multi‑lingual dataset for robust MGMD.  
* **FakeMusicCaps** (Li et al., 2024): text‑conditioned music generation paired with human descriptions.  
* **Cros Vila et al. (2025)**: “in‑the‑wild” Suno/Udio dataset scraped from public feeds.

These datasets reveal a consistent pattern: detectors perform extremely well on seen generators, but generalization to unseen models or manipulated audio remains a major challenge.

## **5.8. Explainability and Human‑Centered Perspectives**

Finally, Cros Vila & Sturm (2025) argue that explainability in AI‑music detection must be treated as a communication problem, not merely a technical one. They show that common XAI tools (e.g., LIME, SHAP, Grad‑CAM) often misalign with human interpretations of musical structure, leading to misleading explanations. This highlights the need for **music‑aware XAI** that can attribute decisions to meaningful musical concepts (e.g., timbre, structure, harmony).

---

## **6\. Bibliography (15–20 References)**

1. **D. Afchar, G. Meseguer-Brocal, and R. Hennequin**, “Detecting music deepfakes is easy but actually hard,” *arXiv preprint arXiv:2405.04181*, 2024\.  
2. **D. Afchar, G. Meseguer-Brocal, and R. Hennequin**, “AI-Generated Music Detection and its Challenges,” in *ICASSP 2025–2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2025, pp. 1–5.  
3. **D. Afchar, G. Meseguer-Brocal, K. Akesbi, and R. Hennequin**, “A Fourier Explanation of AI-music Artifacts,” *arXiv preprint arXiv:2506.19108*, 2025\.  
4. **L. Cros Vila**, *Perspectives on AI and Music: Representation, Detection, and Explanation in the Age of AI-Generated Music*, Ph.D. dissertation, KTH Royal Institute of Technology, 2025\.  
5. **L. Cros Vila, B. Sturm, L. Casini, and D. Dalmazzo**, “The AI Music Arms Race: On the Detection of AI-Generated Music,” *Trans. Int. Soc. Music Inf. Retrieval*, vol. 8, no. 1, pp. 179–194, 2025\.  
6. **E. Dervakos, G. Filandrianos, and G. Stamou**, “Heuristics for evaluation of AI generated music,” in *2020 25th International Conference on Pattern Recognition (ICPR)*, 2021, pp. 9164–9171.  
7. **N. Fišer, M. Á. Martín-Pascual, and C. Andreu-Sánchez**, “Emotional impact of AI-generated vs. human-composed music in audiovisual media: A biometric and self-report study,” *PLOS One*, vol. 20, no. 6, p. e0326498, 2025\.  
8. **M. Frohmann, E. V. Epure, G. Meseguer-Brocal, M. Schedl, and R. Hennequin**, “AI-Generated Song Detection via Lyrics Transcripts,” *arXiv preprint arXiv:2506.18488*, 2025\.  
9. **Y. Kim and S. Go**, “Segment Transformer: AI-Generated Music Detection via Music Structural Analysis,” in *2025 Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)*, 2025, pp. 664–669.  
10. **Y. Kim and S. Go**, “Fusion Segment Transformer: Bi-Directional Attention Guided Fusion Network for AI-Generated Music Detection,” *arXiv preprint arXiv:2601.13647*, 2026\.  
11. **Y. Li, M. Milling, L. Specia, and B. W. Schuller**, “From Audio Deepfake Detection to AI-Generated Music Detection—A Pathway and Overview,” *arXiv preprint arXiv:2412.00571*, 2024\.  
12. **Y. Li, Q. Sun, H. Li, L. Specia, and B. W. Schuller**, “Detecting Machine-Generated Music with Explainability—A Challenge and Early Benchmarks,” *arXiv preprint arXiv:2412.13421*, 2024\.  
13. **Y. Li, H. Li, L. Specia, and B. W. Schuller**, “M6: Multi-generator, Multi-domain, Multi-lingual and cultural, Multi-genres, Multi-instrument Machine-Generated Music Detection Databases,” *arXiv preprint arXiv:2412.06001*, 2024\.  
14. **M. I. Tantra and A. Wicaksana**, “LSTM and CNN-Based Detection of AI-Generated Classical Music From MIDI Features,” *Informatica*, vol. 49, no. 7, 2025\.  
15. **J. Wie, N. Salim, A. A. S. Gunawan, and R. C. Pradana**, “Detecting AI-Generated Music: A Comparative Analysis of Deep Learning Models,” in *2025 Tenth International Conference on Informatics and Computing (ICIC)*, 2025, pp. 1–6.  
16. **L. Cros Vila, B. L. T. Sturm, L. Casini, and D. Dalmazzo**, “The AI Music Arms Race: On the Detection of AI-Generated Music,” Trans. Int. Soc. Music Inf. Retrieval, vol. 8, no. 1, pp. 179–194, 2025\.

---

# **7\. Team Member Objectives**

## **Suchang**

### **Objective 1: Implement the Spectral‑Artifact Detection Module**

* **PI1 (basic):** Compute multi‑resolution STFTs and extract amplitude/phase spectra  
* **PI2 (basic):** Implement Fourier‑peak detection and periodic summation analysis  
* **PI3 (expected):** Reproduce Afchar et al.’s Fourier artifact fingerprints on multiple generators  
* **PI4 (expected):** Build a classifier using spectral fingerprints \+ SSL embeddings  
* **PI5 (advanced):** Develop a generator‑agnostic artifact detector robust to pitch‑shift, noise, and codec compression

**Yuanming**

### **Objective 2: Build the Section‑Level Attribution System**

* **PI1 (basic):** Implement beat-synchronous segmentation and extract CLAP embeddings for individual song sections.  
* **PI2 (basic):** Compute self‑similarity matrices for each track  
* **PI3 (expected):** Train a Segment Transformer for section‑level detection  
* **PI4 (expected):** Visualize section-level predictions using timelines and confusion matrices.  
* **PI5 (advanced):** Analyze whether section-level detection correlates with musical structure or with non-musical artifacts (e.g., sampling rate, spectral bandwidth).

