# Research Proposal 1: Trustworthy and Resource-Efficient AI for Early Cancer Screening in Underserved Clinical Settings

## Background and Motivation
Early detection is a critical component of effective cancer care, but medical-image screening can depend on specialist expertise, appropriate imaging infrastructure, and timely interpretation. This project proposes lightweight, explainable, and uncertainty-aware vision transformer architectures to support cancer screening and triage in resource-constrained clinical environments. The objective is to support clinicians rather than replace them.

## Research Problem
Many medical-image AI systems have substantial computational requirements, limited interpretability, unreliable confidence estimates, and reduced performance under deployment conditions that differ from their training data. The project will investigate whether these limitations can be addressed jointly.

## Objectives
- Develop lightweight vision transformer architectures for medical-image classification.
- Investigate oral and/or cervical disease screening where suitable datasets are available.
- Incorporate explainable-AI mechanisms to identify image regions contributing to predictions.
- Develop calibrated confidence estimates and selective prediction.
- Evaluate robustness across heterogeneous image sources.
- Reduce memory and computational requirements for practical deployment.

## Research Questions
1. Can lightweight vision transformers retain competitive screening performance with substantially lower computational cost?
2. Can calibrated uncertainty distinguish reliable predictions from cases requiring specialist review?
3. Can explainability methods provide meaningful localization of features influencing predictions?
4. How robust are the models across heterogeneous datasets and imaging conditions?
5. Can efficiency, explainability, and uncertainty be integrated into a practical AI-assisted screening framework?

## Methodology
The study will use publicly available or appropriately authorized medical-imaging datasets. CNN and standard vision-transformer baselines will be compared with lightweight architectures using efficient attention, feature compression, distillation, pruning, or related techniques. Evaluation will include sensitivity, specificity, F1, AUROC, precision-recall measures, parameter count, FLOPs, memory, and inference latency. Calibration can be evaluated using Expected Calibration Error and Brier score. A selective-prediction mechanism will abstain or refer cases when uncertainty exceeds a validated threshold.

## Expected Contributions
- A computationally efficient medical-vision architecture.
- Integrated explainability and uncertainty calibration.
- Evidence on reliability across heterogeneous medical-image conditions.
- A selective screening strategy for identifying cases requiring expert review.
- Reproducible methods supporting trustworthy medical AI in resource-constrained settings.

## NIW National-Importance Positioning
The proposed work is designed to produce generalizable methods rather than a proprietary solution for one employer or clinic. Its prospective importance rests on improving the technical foundations for accessible, reliable, and scalable AI-assisted disease screening. For an NIW filing, claims concerning U.S. screening disparities, specialist access, rural healthcare, and federal priorities should be supported with current authoritative evidence rather than asserted solely from this proposal.
