# CRVQ: Channel-Relaxed Vector Quantization <br> for Extreme Compression of LLMs

[![Paper](https://img.shields.io/badge/Paper-TACL-b31b1b.svg)](https://direct.mit.edu/tacl/article/doi/10.1162/TACL.a.45/133863)
[![Conference](https://img.shields.io/badge/Conference-EMNLP%202025-blue)](https://transacl.org/index.php/tacl/article/view/8165)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

**Yuzhuang Xu**, Shiyu Ji, Qingfu Zhu, Wanxiang Che

*Harbin Institute of Technology (HIT)*

</div>

---

## 📖 Introduction

Welcome to the official repository for **CRVQ** (Channel-Relaxed Vector Quantization).

As Large Language Models (LLMs) continue to grow in size, deploying them on resource-constrained devices has become a significant challenge. Existing Post-Training Quantization (PTQ) methods often suffer from severe performance degradation when pushing compression below 2 bits.

Model deployment is achieved by compressing LLMs to extremely low-bit widths. Vector quantization is a promising recent approach that primarily achieves capability scaling by scaling the codebook's capacity.

**CRVQ** is a novel quantization framework designed to break the weakness of VQ. By introducing a hardware-friendly "Channel-Relaxed" mechanism, CRVQ enables LLMs to maintain high performance even at extreme compression rates (approaching 1-bit), instead of scaling the width of codes.

### ✨ Core Features & Highlights

* **🏆 Good Performance**: CRVQ achieves a massive **38.9% performance improvement** over strong PTQ baselines in sub-2-bit settings.
* **🧠 Channel-Relaxed Mechanism**: 
    * **Critical Channel Reordering**: Intelligently identifies and rearranges channels that are sensitive to quantization noise.
    * **Extended Codebooks**: Relaxes constraints on critical channels by assigning them larger codebooks, significantly reducing error.
* **⚡ Viable 1-bit Compression**: Makes near 1-bit compression practical for real-world LLM deployment for the first time.
* **🛠 Flexible Bit-Widths**: Supports arbitrary bit-width configurations to balance model size and accuracy.

---

## 🚀 Method Overview

The core innovation of CRVQ is **Channel Relaxation**. Instead of treating all weights equally, we decouple the quantization burden based on channel importance.

1.  **Sensitivity Analysis**: We measure the impact of each channel on the final output.
2.  **Relaxation & Quantization**: "Critical" channels are quantized with higher fidelity (via extended codebooks or reordering), while non-critical channels undergo extreme compression.

---

## 📊 Experimental Results

We evaluated CRVQ on various LLM families (LLaMA, Phi, Qwen) across multiple benchmarks. The main results are as follows:

| Model | Method | Wbits | PPL-wiki2 | PPL-c4 | Avg-Acc |
| :--- | :--- | :---: | :---: | :---: | :---: |
| LLaMA2-13B | FP16 | 16.0 | 4.57 | 6.05 | 73.25 |
| | BiLLM | 1.08 | 20.52 | 32.01 | 48.71 |
| | AQLM | 1.01 | 15.25 | 18.35 | 44.78 |
| | **CRVQ** | 1.06 | **9.81** | **12.48** | **53.30** |

> *Note: For detailed results on Zero-shot tasks and other model sizes, please refer to our [Paper](https://direct.mit.edu/tacl/article/doi/10.1162/TACL.a.45/133863).*

---

## 🛠️ Installation

```bash
# 1. Clone the repository
git clone [https://github.com/xuyuzhuang11/CRVQ.git](https://github.com/xuyuzhuang11/CRVQ.git)
cd CRVQ

# 2. Create a virtual environment (Recommended)
conda create -n crvq python=3.10
conda activate crvq

# 3. Install dependencies
pip install -r requirements.txt

```

---

## 💻 Usage

### 1. Data Preparation

Ensure you have the calibration dataset ready (**Redpajama**).

### 2. Quantization (Run CRVQ)

Run the main script to quantize a model. The script handles channel reordering and vector quantization automatically.

```bash
bash run_CRVQ.sh
bash finetune.sh

```

### 3. Evaluation

Evaluate the quantized model's Perplexity (PPL) or Zero-shot accuracy:

```bash
bash ppl_eval.sh
bash lm_eval.sh

```

---

## 📝 Citation

If you find our work or code useful for your research, please consider citing our TACL paper:

```bibtex
@article{xu2025crvq,
  title={{CRVQ}: Channel-Relaxed Vector Quantization for Extreme Compression of {LLMs}},
  author={Xu, Yuzhuang and Ji, Shiyu and Zhu, Qingfu and Che, Wanxiang},
  journal={Transactions of the Association for Computational Linguistics (TACL)},
  volume={13},
  pages={1488-1506},
  year={2025},
  url = {https://doi.org/10.1162/TACL.a.45}
}

```

---

## 📧 Contact

For any questions or suggestions, please feel free to open an issue or contact the authors:

* **Yuzhuang Xu**: [xyz@ir.hit.edu.cn](mailto:xyz@ir.hit.edu.cn)
