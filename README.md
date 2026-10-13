# Indic Multi-Task Natural Language Understanding (NLU) System

A lightweight, local PyTorch Proof of Concept (PoC) for multi-task Natural Language Understanding across Indic languages (Hindi, Bengali, etc.) using `ai4bharat/indic-bert`.

The model jointly solves two core NLU tasks using a shared Transformer encoder and decoupled task-specific classification heads:
1. **Sentiment Analysis** (3 classes: `Positive`, `Neutral`, `Negative`)
2. **Intent Classification** (3 classes: `Inquiry`, `Opinion`, `Factual`)

---

## Architecture Overview

```
                      ┌───────────────────────────────────────┐
                      │  Input Sequence (input_ids, mask)     │
                      └──────────────────┬────────────────────┘
                                         ▼
                      ┌───────────────────────────────────────┐
                      │    ai4bharat/indic-bert Encoder       │
                      │ (~33.44M params, 12 ALBERT layers)    │
                      └──────────────────┬────────────────────┘
                                         ▼
                      ┌───────────────────────────────────────┐
                      │   [CLS] Token Extraction (Index 0)    │
                      │         Shape: [batch_size, 768]      │
                      └──────────────────┬────────────────────┘
                                         ▼
                      ┌───────────────────────────────────────┐
                      │         Dropout (p = 0.2)             │
                      └──────────────┬────────────────┬───────┘
                                     │                │
            ┌────────────────────────┴─┐            ┌─┴────────────────────────┐
            ▼                          ▼            ▼                          ▼
┌─────────────────────────┐          ┌─────────────────────────┐
│     sentiment_head      │          │       intent_head       │
│  nn.Linear(768, 3)      │          │   nn.Linear(768, 3)     │
└───────────┬─────────────┘          └───────────┬─────────────┘
            ▼                                    ▼
┌─────────────────────────┐          ┌─────────────────────────┐
│       sent_logits       │          │      intent_logits      │
│  Shape: [batch_size, 3] │          │  Shape: [batch_size, 3] │
└─────────────────────────┘          └─────────────────────────┘
```

---

## Key Features

- **Indic-BERT Diacritic Preservation**: Initialized with `keep_accents=True` to preserve vital Indic orthographic markers, vowel signs (*matras*), nuktas, and viramas/halants.
- **System-Level UTF-8 & Windows Console Protection**: Automatic standard I/O stream reconfiguration (`sys.stdin.reconfigure`, `os.system("chcp 65001")`) preventing character mojibake and encoding crashes on Windows and Linux terminals.
- **Unicode NFC Normalization**: Normalizes all input strings to canonical composition (`NFC`), stripping invisible zero-width spaces (`\u200b`) and Byte Order Marks (`\ufeff`).
- **Composite Multi-Task Loss**: Joint optimization with $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{sentiment}} + \mathcal{L}_{\text{intent}}$ using `AdamW` (`lr=2e-5`, `weight_decay=0.01`) and gradient clipping (`max_norm=1.0`).
- **Multi-Modal Testing Interface**:
  - **Mode A (Numbered Menu)**: Pre-configured test cases across Hindi and Bengali selectable by number keys.
  - **Mode B (File Reader)**: Batch inference from UTF-8 text files (`--file <path>`).
  - **Mode C (Interactive Mode)**: Live input loop with independent Softmax probability breakdowns and unescaped JSON output.

---

## Installation

Ensure Python 3.10+ is installed:

```bash
git clone https://github.com/tathagata-dey-debug/Indic-NLU-system.git
cd Indic-NLU-system
pip install -r requirements.txt
```

---

## Usage

### 1. Run Interactive CLI & Automated Verification
```bash
python indic_nlu_poc.py
```
This runs automated unit assertions, executes a 3-epoch training sanity loop, runs inference on sample sentences, and enters the interactive CLI menu:
- Press `1` to `5` to test pre-configured sentences without terminal clipboard issues.
- Type `custom` to test your own sentence.
- Type `exit` to quit.

### 2. Batch Inference from a Text File
```bash
python indic_nlu_poc.py --file test_indic_samples.txt
```

### 3. Programmatic Usage in Python
```python
from indic_nlu_poc import IndicMultiTaskNLU, predict_indic_text, tokenizer, device

model = IndicMultiTaskNLU().to(device)

result = predict_indic_text(
    text="क्या यह फोन 5G सपोर्ट करता है?",
    model=model,
    tokenizer=tokenizer,
    device=device
)

print(result["sentiment"]["label"], result["sentiment"]["confidence"])
print(result["intent"]["label"], result["intent"]["confidence"])
```

---

## License
MIT License
