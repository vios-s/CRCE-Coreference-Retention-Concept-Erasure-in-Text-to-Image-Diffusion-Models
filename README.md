# CRCE: Coreference Retention Concept Erasure in Text-to-Image Diffusion Models

[![Paper](http://img.shields.io/badge/paper-arxiv.2503.14232-B31B1B.svg)](https://arxiv.org/abs/2503.14232)
[![BMVC](http://img.shields.io/badge/BMVC-2025-4b44ce.svg)](https://openreview.net/forum?id=7bILZDwb1c)

**CRCE** is a novel concept erasure framework for text-to-image diffusion models that handles coreferential concepts (synonyms, related terms) to prevent bypass attacks while preserving model utility.

## Table of Contents

- [CRCE: Coreference Retention Concept Erasure in Text-to-Image Diffusion Models](#crce-coreference-retention-concept-erasure-in-text-to-image-diffusion-models)
  - [Table of Contents](#table-of-contents)
  - [Method](#method)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Multi-Agent Framework](#multi-agent-framework)
  - [CorefConcept Dataset](#corefconcept-dataset)
  - [Results](#results)
  - [Citation](#citation)
  - [License](#license)

## Method

CRCE uses a multi-objective loss function:
```
L_total = L_anchor + α·L_coref + β·L_retain
```
- **L_anchor**: Erases target concept
- **L_coref**: Erases related concepts (synonyms, variations)  
- **L_retain**: Preserves unrelated concepts

## Installation

```bash
git clone https://github.com/vios-s/CRCE-Coreference-Retention-Concept-Erasure-in-Text-to-Image-Diffusion-Models
cd CRCE-Coreference-Retention-Concept-Erasure-in-Text-to-Image-Diffusion-Models
pip install -r requirements.txt
```

## Usage

```python
from ours_tools import execute_ours_unlearn

# Erase "airplane" and related concepts while preserving other flying objects
result = execute_ours_unlearn(
    erase_concept="airplane",
    coref_concept="aeroplane,plane,jet plane,passenger plane",
    retain_concept="hot air balloon,blimp,rocket,drone",
    iterations=500,
    train_method='xattn-strict'
)
```

## Multi-Agent Framework

For automated experiments with LLM-guided concept identification:
```bash
python main.py  # Requires llmconfig.json with API keys
```

## CorefConcept Dataset

Pre-curated concept sets for reproducible experiments:
- `CorefConcept/object.json` - CIFAR-10 based objects
- `CorefConcept/celebrity.json` - Public figures  
- `CorefConcept/ip.json` - Intellectual property

## Results

CRCE achieves 95%+ target concept removal while maintaining 85%+ retention quality, with 2-5x faster training compared to full model fine-tuning.

## Citation

```bibtex
@inproceedings{xue2025crce,
  title={CRCE: Coreference Retention Concept Erasure in Text-to-Image Diffusion Models},
  author={Xue, Yuyang and Moroshko, Edward and Chen, Feng and Sun, Jingyu and McDonagh, Steven and Tsaftaris, Sotirios A},
  booktitle={British Machine Vision Conference (BMVC)},
  year={2025}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

