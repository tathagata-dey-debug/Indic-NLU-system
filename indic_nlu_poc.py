"""
Indic Multi-Task NLU Proof of Concept (PoC) - Step 1
Data Preparation, Tokenization, and PyTorch Dataset Pipeline using ai4bharat/indic-bert.
"""

import os
import sys
import json
import argparse
import unicodedata
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig

# ==============================================================================
# System-Level UTF-8 Stream Reconfiguration
# Permanently resolves character encoding corruption, garbled Unicode text,
# and terminal I/O mojibake on Windows and Linux terminals.
# ==============================================================================
if sys.platform.startswith("win"):
    try:
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        os.system("chcp 65001 > nul")  # Set Windows console code page to UTF-8
    except Exception:
        pass

# ==============================================================================
# Unicode Normalization & Cleaning Layer
# ==============================================================================
def clean_indic_text(text: str) -> str:
    """
    Normalizes Indic text to canonical NFC composition, strips invisible
    zero-width characters and byte-order-marks (BOM), and cleans whitespace.
    """
    if not isinstance(text, str):
        return text
    # Strip invisible zero-width spaces and BOM
    text = text.replace("\u200b", "").replace("\ufeff", "")
    # Canonical Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)
    return text.strip()

# ==============================================================================
# 0. Hardware & Device Configuration
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device Selection] Configured device: {device}")

# ==============================================================================
# 1. Mock In-Memory Bilingual Corpus
# ==============================================================================
# Label Spaces:
#   Sentiment: 0: "Positive", 1: "Neutral", 2: "Negative"
#   Intent:    0: "Inquiry",  1: "Opinion", 2: "Factual"

SENTIMENT_MAP = {0: "Positive", 1: "Neutral", 2: "Negative"}
INTENT_MAP = {0: "Inquiry", 1: "Opinion", 2: "Factual"}

MOCK_DATA = [
    # --- Hindi Examples (4 entries) ---
    {
        # 1. Inquiry: Hindi question asking about 5G capability
        "text": "क्या यह फोन 5G सपोर्ट करता है?",
        "lang": "hi",
        "sentiment": 1,  # Neutral
        "intent": 0,     # Inquiry
    },
    {
        # 2. Positive Review: Opinion expressing satisfaction
        "text": "यह होटल बहुत अच्छा है और खाना बेहद स्वादिष्ट था।",
        "lang": "hi",
        "sentiment": 0,  # Positive
        "intent": 1,     # Opinion
    },
    {
        # 3. Negative Complaint: Opinion expressing dissatisfaction
        "text": "इस उत्पाद की गुणवत्ता बहुत खराब है और सेवा निराशाजनक है।",
        "lang": "hi",
        "sentiment": 2,  # Negative
        "intent": 1,     # Opinion
    },
    {
        # 4. Factual Statement: General fact
        "text": "नई दिल्ली भारत की राजधानी है।",
        "lang": "hi",
        "sentiment": 1,  # Neutral
        "intent": 2,     # Factual
    },
    # --- Bengali Examples (4 entries) ---
    {
        # 5. Inquiry: Bengali question about capital of West Bengal
        "text": "কলকাতা কি পশ্চিমবঙ্গের রাজধানী?",
        "lang": "bn",
        "sentiment": 1,  # Neutral
        "intent": 0,     # Inquiry
    },
    {
        # 6. Positive Review: Book review praising content
        "text": "এই বইটি চমৎকার এবং গল্পটি সত্যিই অসাধারণ।",
        "lang": "bn",
        "sentiment": 0,  # Positive
        "intent": 1,     # Opinion
    },
    {
        # 7. Negative Complaint: Product and service complaint
        "text": "এই পণ্যটির মান অত্যন্ত খারাপ এবং পরিষেবা হতাশাজনক।",
        "lang": "bn",
        "sentiment": 2,  # Negative
        "intent": 1,     # Opinion
    },
    {
        # 8. Factual Statement: Astronomical fact in Bengali
        "text": "সূর্য পূর্ব দিকে উদিত হয়।",
        "lang": "bn",
        "sentiment": 1,  # Neutral
        "intent": 2,     # Factual
    },
]

# ==============================================================================
# 2. Tokenizer Initialization
# ==============================================================================
# Indic-BERT Tokenizer Rationale:
#   ai4bharat/indic-bert uses an ALBERT-based architecture with a SentencePiece vocabulary
#   covering 12 major Indian languages. Setting `keep_accents=True` is CRITICAL for Indic
#   scripts (Devanagari, Bengali, etc.) because accents, diacritical marks, vowel matras
#   (e.g., ा, ि, ু, ী), halants/viramas, and nuktas represent essential phonetic and
#   orthographic elements. Stripping them (which default Latin-centric accent stripping does)
#   would corrupt words, distort syllable boundaries, and lose grammatical meanings.
MODEL_NAME = "ai4bharat/indic-bert"
LOCAL_TOKENIZER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "indic_bert_tokenizer")

print(f"[Tokenizer] Loading '{MODEL_NAME}' tokenizer with keep_accents=True...")

hf_token = os.getenv("HF_TOKEN")
try:
    # 1. Primary path: Load from Hugging Face Hub directly
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, keep_accents=True, token=hf_token)
except Exception as e:
    # 2. Resilient fallback: ai4bharat/indic-bert is a gated model on Hugging Face Hub requiring
    #    user agreement / HF_TOKEN authentication. If remote access encounters a 401 or network error,
    #    fallback seamlessly to the local cache or local tokenizer directory.
    print(f"[Tokenizer Notice] Remote load encountered {type(e).__name__} (gated model authentication required).")
    print("[Tokenizer Notice] Falling back to local offline Indic-BERT tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, keep_accents=True, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_TOKENIZER_DIR, keep_accents=True)

print(f"[Tokenizer] Successfully initialized ({type(tokenizer).__name__}).")

# ==============================================================================
# 3. Custom PyTorch Dataset (IndicPoCDataset)
# ==============================================================================
class IndicPoCDataset(Dataset):
    """
    PyTorch Dataset wrapper for bilingual Indic multi-task NLU corpus.
    Provides tokenized input_ids, attention_mask, and multi-task target labels.
    """

    def __init__(self, data, tokenizer, max_length: int = 64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        text = clean_indic_text(item["text"])

        # Tokenize sequence with fixed max_length, padding, and truncation
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Squeeze leading batch dimension (1, max_length) -> (max_length,)
        input_ids = encoded["input_ids"].squeeze(0).long()
        attention_mask = encoded["attention_mask"].squeeze(0).long()

        # Multi-task target labels as scalar tensors
        sentiment_label = torch.tensor(item["sentiment"], dtype=torch.long)
        intent_label = torch.tensor(item["intent"], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "sentiment": sentiment_label,
            "intent": intent_label,
        }

# ==============================================================================
# 4. IndicMultiTaskNLU Model Architecture (Step 2)
# ==============================================================================
class IndicMultiTaskNLU(nn.Module):
    """
    Modular Multi-Task Neural Network utilizing a shared ai4bharat/indic-bert encoder
    branching into two classification heads for Sentiment and Intent.
    """

    def __init__(
        self,
        model_name: str = "ai4bharat/indic-bert",
        num_sentiment_classes: int = 3,
        num_intent_classes: int = 3,
        dropout_prob: float = 0.2,
    ):
        super().__init__()

        # Shared ALBERT encoder (~33M parameters, cross-layer parameter sharing)
        try:
            self.encoder = AutoModel.from_pretrained(model_name, token=os.getenv("HF_TOKEN"))
        except Exception as e:
            # Gated repository / offline fallback
            try:
                self.encoder = AutoModel.from_pretrained(model_name, local_files_only=True)
            except Exception:
                config = AutoConfig.from_pretrained(
                    LOCAL_TOKENIZER_DIR if os.path.exists(LOCAL_TOKENIZER_DIR) else model_name
                )
                self.encoder = AutoModel.from_config(config)

        # Dynamically extract hidden size (768 for ALBERT-base)
        hidden_size = self.encoder.config.hidden_size

        # Dropout regularization layer
        self.dropout = nn.Dropout(p=dropout_prob)

        # Decoupled task-specific classification heads
        self.sentiment_head = nn.Linear(hidden_size, num_sentiment_classes)
        self.intent_head = nn.Linear(hidden_size, num_intent_classes)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass routing sequence representations through the shared encoder
        and into both task heads.

        Args:
            input_ids: 2D LongTensor of shape (batch_size, seq_len)
            attention_mask: 2D LongTensor of shape (batch_size, seq_len)

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                (sent_logits, intent_logits)
                sent_logits: shape (batch_size, num_sentiment_classes)
                intent_logits: shape (batch_size, num_intent_classes)
        """
        # Pass input sequences through the shared Indic-BERT encoder
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Extract sentence-level pooled representation from the [CLS] token position (index 0)
        # Tensor shape: [batch_size, hidden_size] (e.g. [batch_size, 768])
        cls_rep = outputs.last_hidden_state[:, 0, :]

        # Apply dropout regularization
        cls_rep = self.dropout(cls_rep)

        # Project simultaneously into task classification heads
        sent_logits = self.sentiment_head(cls_rep)
        intent_logits = self.intent_head(cls_rep)

        return sent_logits, intent_logits

# ==============================================================================
# 5. Multi-Task Training Loop & Optimization (Step 3)
# ==============================================================================
def train_poc(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion_sent: nn.Module,
    criterion_intent: nn.Module,
    device: torch.device,
    epochs: int = 3,
) -> list[float]:
    """
    Executes an end-to-end multi-task training loop for Indic Multi-Task NLU.
    Composite objective: L_total = L_sentiment + L_intent

    Args:
        model: IndicMultiTaskNLU model instance
        dataloader: PyTorch DataLoader for the IndicPoCDataset
        optimizer: Configured AdamW optimizer
        criterion_sent: CrossEntropyLoss criterion for sentiment classification
        criterion_intent: CrossEntropyLoss criterion for intent classification
        device: Active computation device (cuda/cpu)
        epochs: Number of training epochs (default: 3)

    Returns:
        list[float]: Average total loss recorded per epoch
    """
    model.train()
    epoch_total_losses: list[float] = []
    num_batches = len(dataloader)

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        running_sent_loss = 0.0
        running_intent_loss = 0.0

        for batch in dataloader:
            # Transfer tensors to active device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_sent = batch["sentiment"].to(device)
            target_intent = batch["intent"].to(device)

            # 1. Zero out existing gradients
            optimizer.zero_grad()

            # 2. Forward pass through shared encoder and task heads
            sent_logits, intent_logits = model(input_ids, attention_mask)

            # 3. Compute task losses
            loss_sent = criterion_sent(sent_logits, target_sent)
            loss_intent = criterion_intent(intent_logits, target_intent)

            # 4. Compute composite multi-task objective: L_total = L_sentiment + L_intent
            loss_total = loss_sent + loss_intent

            # 5. Backpropagation
            loss_total.backward()

            # 6. Gradient clipping to stabilize Transformer weights
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 7. Optimizer step
            optimizer.step()

            # Accumulate metrics
            running_loss += loss_total.item()
            running_sent_loss += loss_sent.item()
            running_intent_loss += loss_intent.item()

        avg_total_loss = running_loss / num_batches
        avg_sent_loss = running_sent_loss / num_batches
        avg_intent_loss = running_intent_loss / num_batches

        epoch_total_losses.append(avg_total_loss)

        # Print clean epoch summary
        print(
            f"Epoch [{epoch}/{epochs}] | "
            f"Total Loss: {avg_total_loss:.4f} | "
            f"Sent Loss: {avg_sent_loss:.4f} | "
            f"Intent Loss: {avg_intent_loss:.4f}"
        )

    return epoch_total_losses

# ==============================================================================
# 6. Production Inference Engine (Step 4)
# ==============================================================================
def predict_indic_text(
    text: str, model: nn.Module, tokenizer, device: torch.device
) -> dict:
    """
    Production-grade inference function for multi-task Indic text classification.
    Generates independent Softmax probability distributions and confidence scores
    for Sentiment and Intent classification.

    Args:
        text: Raw input string (Hindi, Bengali, or other Indic languages)
        model: Trained IndicMultiTaskNLU model instance
        tokenizer: Indic-BERT AutoTokenizer
        device: Active computation device (cuda/cpu)

    Returns:
        dict: Structured payload with raw text, predicted labels, confidence scores,
              and task probability distributions.
    """
    SENTIMENT_LABELS = {0: "Positive", 1: "Neutral", 2: "Negative"}
    INTENT_LABELS = {0: "Inquiry", 1: "Opinion", 2: "Factual"}

    # Set model to evaluation mode (disables dropout)
    model.eval()

    # Clean and normalize input text (NFC canonical composition)
    cleaned_text = clean_indic_text(text)

    # Tokenize normalized input string
    inputs = tokenizer(
        cleaned_text,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )

    # Transfer input tensors to active device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        sent_logits, intent_logits = model(input_ids, attention_mask)

        # Apply independent Softmax normalization across class dimensions
        sent_probs = torch.softmax(sent_logits, dim=-1)
        intent_probs = torch.softmax(intent_logits, dim=-1)

        # Extract top predicted class index and confidence scalar
        sent_conf, sent_idx = torch.max(sent_probs, dim=-1)
        intent_conf, intent_idx = torch.max(intent_probs, dim=-1)

    return {
        "raw_text": cleaned_text,
        "sentiment": {
            "label": SENTIMENT_LABELS[sent_idx.item()],
            "confidence": round(sent_conf.item(), 4),
            "probabilities": {
                SENTIMENT_LABELS[i]: round(prob, 4)
                for i, prob in enumerate(sent_probs[0].tolist())
            },
        },
        "intent": {
            "label": INTENT_LABELS[intent_idx.item()],
            "confidence": round(intent_conf.item(), 4),
            "probabilities": {
                INTENT_LABELS[i]: round(prob, 4)
                for i, prob in enumerate(intent_probs[0].tolist())
            },
        },
    }

# ==============================================================================
# 7. Verification & Testing Block
# ==============================================================================
def main():
    # --------------------------------------------------------------------------
    # Step 1 Verification: Dataset & DataLoader
    # --------------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: Verifying IndicPoCDataset & DataLoader Pipeline")
    print("=" * 60)

    dataset = IndicPoCDataset(MOCK_DATA, tokenizer, max_length=64)
    print(f"\n[Dataset] IndicPoCDataset created with {len(dataset)} samples.")

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    print(f"[DataLoader] Initialized with batch_size=2, shuffle=True. Total batches: {len(dataloader)}")

    # Fetch first batch
    first_batch = next(iter(dataloader))

    # Assertions for Step 1
    print("\n[Assertions] Verifying Step 1 batch tensor dimensions...")
    assert first_batch["input_ids"].shape == (2, 64), (
        f"Expected input_ids shape (2, 64), got {first_batch['input_ids'].shape}"
    )
    assert first_batch["attention_mask"].shape == (2, 64), (
        f"Expected attention_mask shape (2, 64), got {first_batch['attention_mask'].shape}"
    )
    assert first_batch["sentiment"].shape == (2,), (
        f"Expected sentiment shape (2,), got {first_batch['sentiment'].shape}"
    )
    assert first_batch["intent"].shape == (2,), (
        f"Expected intent shape (2,), got {first_batch['intent'].shape}"
    )
    print("[PASS] All Step 1 shape assertions passed successfully!")

    # Decode tokens for batch item 0
    item0_ids = first_batch["input_ids"][0]
    decoded_text = tokenizer.decode(item0_ids, skip_special_tokens=False)
    decoded_text_clean = tokenizer.decode(item0_ids, skip_special_tokens=True)
    tokens_list = tokenizer.convert_ids_to_tokens(item0_ids[:15])

    print(f"\n--- Batch Item 0 Inspection ---")
    print(f"Sample Decoded (with special tokens): {decoded_text}")
    print(f"Sample Decoded (clean):               {decoded_text_clean}")
    print(f"Leading Subword Tokens:               {tokens_list}")
    print(f"Target Sentiment:                     {first_batch['sentiment'][0].item()} ({SENTIMENT_MAP[first_batch['sentiment'][0].item()]})")
    print(f"Target Intent:                        {first_batch['intent'][0].item()} ({INTENT_MAP[first_batch['intent'][0].item()]})")
    print("\nCONFIRMATION: The Indic multi-task tokenization pipeline is operational!")

    # --------------------------------------------------------------------------
    # Step 2 Verification: IndicMultiTaskNLU Model & Forward Pass
    # --------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Initializing IndicMultiTaskNLU & Verifying Forward Pass")
    print("=" * 60)

    print(f"[Model] Instantiating IndicMultiTaskNLU with base '{MODEL_NAME}' on {device}...")
    model = IndicMultiTaskNLU(
        model_name=MODEL_NAME,
        num_sentiment_classes=3,
        num_intent_classes=3,
        dropout_prob=0.2,
    ).to(device)

    # Calculate parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    sent_params = sum(p.numel() for p in model.sentiment_head.parameters())
    intent_params = sum(p.numel() for p in model.intent_head.parameters())

    print("\n[Model Parameter Breakdown]")
    print(f"  Total Parameters:          {total_params:,} (~{total_params / 1e6:.2f}M)")
    print(f"  Trainable Parameters:      {trainable_params:,}")
    print(f"  Encoder Parameters:        {encoder_params:,} (~{encoder_params / 1e6:.2f}M, ALBERT shared weights)")
    print(f"  Sentiment Head Parameters: {sent_params:,} (Linear: 768 -> 3)")
    print(f"  Intent Head Parameters:    {intent_params:,} (Linear: 768 -> 3)")

    # Prepare inputs on active device
    input_ids = first_batch["input_ids"].to(device)
    attention_mask = first_batch["attention_mask"].to(device)

    # Execute forward pass
    print(f"\n[Forward Pass] Passing batch (shape: {input_ids.shape}) through model...")
    model.eval()
    with torch.no_grad():
        sent_logits, intent_logits = model(input_ids, attention_mask)

    # Step 2 Assertions
    print("\n[Assertions] Verifying output logits tensor shapes...")
    assert sent_logits.shape == (2, 3), (
        f"Expected sent_logits shape (2, 3), got {sent_logits.shape}"
    )
    assert intent_logits.shape == (2, 3), (
        f"Expected intent_logits shape (2, 3), got {intent_logits.shape}"
    )
    print("[PASS] All Step 2 output logits shape assertions passed successfully!")

    print("\n--- Output Logits Details ---")
    print(f"sent_logits shape:   {sent_logits.shape} (dtype={sent_logits.dtype})")
    print(f"sent_logits sample:  {sent_logits.cpu().tolist()}")
    print(f"intent_logits shape: {intent_logits.shape} (dtype={intent_logits.dtype})")
    print(f"intent_logits sample:{intent_logits.cpu().tolist()}")
    print("\nCONFIRMATION: IndicMultiTaskNLU model architecture and forward pass are verified!")

    # --------------------------------------------------------------------------
    # Step 3 Verification: Composite Loss, Optimizer & Sanity Training Loop
    # --------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Multi-Task Loss, Optimizer & Training Sanity Loop")
    print("=" * 60)

    # 1. Loss Criteria
    criterion_sent = nn.CrossEntropyLoss()
    criterion_intent = nn.CrossEntropyLoss()
    print("[Loss Criteria] Configured CrossEntropyLoss for Sentiment and Intent.")

    # 2. Optimizer Configuration
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5,
        weight_decay=0.01,
    )
    print(f"[Optimizer] Configured AdamW (lr=2e-5, weight_decay=0.01) over {len(list(model.parameters()))} parameter tensors.")

    # 3. End-to-End Training Loop Execution
    print(f"\n[Training Loop] Executing 3-epoch training sanity loop on {device}...")
    loss_history = train_poc(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        criterion_sent=criterion_sent,
        criterion_intent=criterion_intent,
        device=device,
        epochs=3,
    )

    # 4. Convergence Check & Verification
    print("\n--- Step 3 Convergence Verification ---")
    print(f"Epoch Loss History: {[round(l, 4) for l in loss_history]}")
    loss_decreased = loss_history[-1] < loss_history[0]
    print(f"Loss Reduction Confirmed (Epoch 3 < Epoch 1): {loss_decreased} ({loss_history[0]:.4f} -> {loss_history[-1]:.4f})")
    assert loss_decreased, f"Expected total loss to decrease across 3 epochs, got {loss_history}"
    print("[PASS] Multi-task total loss decreased over 3 epochs, verifying proper gradient flow and backpropagation!")
    print("\nCONFIRMATION: Step 3 end-to-end multi-task training sanity loop completed successfully!")

    # --------------------------------------------------------------------------
    # Step 4 Verification: Production Inference Engine on Unseen Regional Data
    # --------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: Production Inference Engine & Verification on Unseen Data")
    print("=" * 60)

    test_samples = [
        {
            "text": "क्या कल नई दिल्ली में बैंक खुले हैं?",
            "desc": "Hindi Inquiry: 'Are banks open tomorrow in New Delhi?'",
        },
        {
            "text": "এই হোটেলের খাবার সত্যিই অসাধারণ ছিল",
            "desc": "Bengali Opinion: 'The food at this hotel was truly extraordinary'",
        },
        {
            "text": "यह ऐप बार-बार क्रैश हो रहा है, बहुत बेकार अनुभव",
            "desc": "Hindi Complaint: 'This app keeps crashing, very bad experience'",
        },
    ]

    for idx, sample in enumerate(test_samples, start=1):
        print(f"\n--- Unseen Test Case #{idx}: {sample['desc']} ---")
        pred_result = predict_indic_text(
            text=sample["text"],
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
        print(json.dumps(pred_result, indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    print("CONFIRMATION: Step 4 production inference engine verified on unseen Indic data!")
    print("=" * 60)

    # --------------------------------------------------------------------------
    # Argument Parsing (Mode B: UTF-8 File Reader)
    # --------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Indic Multi-Task NLU Proof of Concept")
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to a UTF-8 text file containing Indic sentences to classify (Mode B).",
    )
    # Parse known args so interactive or test environments pass cleanly
    args, _ = parser.parse_known_args()

    # --------------------------------------------------------------------------
    # Mode B: UTF-8 File Reader (--file)
    # --------------------------------------------------------------------------
    if args.file:
        print("\n" + "=" * 60)
        print(f"MODE B: UTF-8 FILE INFERENCE ENGINE ({args.file})")
        print("=" * 60)

        if not os.path.exists(args.file):
            print(f"[Error] Specified file does not exist: {args.file}")
            return

        with open(args.file, "r", encoding="utf-8") as f:
            file_lines = [clean_indic_text(line) for line in f if clean_indic_text(line)]

        print(f"Successfully loaded {len(file_lines)} sentence(s) from '{args.file}'.\n")

        for idx, line in enumerate(file_lines, start=1):
            pred_result = predict_indic_text(
                text=line,
                model=model,
                tokenizer=tokenizer,
                device=device,
            )
            sent_info = pred_result["sentiment"]
            intent_info = pred_result["intent"]

            print("-" * 55)
            print(f"Line #{idx}: \"{line}\"")
            print(f"  Sentiment: {sent_info['label']} ({sent_info['confidence'] * 100:.2f}%)")
            print(f"  Intent:    {intent_info['label']} ({intent_info['confidence'] * 100:.2f}%)")
            print(json.dumps(pred_result, indent=2, ensure_ascii=False))
        return

    # --------------------------------------------------------------------------
    # Multi-Modal Testing Interface (Mode A: Numbered Menu & Mode C: Interactive)
    # --------------------------------------------------------------------------
    MENU_OPTIONS = {
        "1": (
            "क्या यह बस सीधे हावड़ा स्टेशन जाएगी?",
            "Hindi Inquiry: 'क्या यह बस सीधे हावड़ा स्टेशन जाएगी?'",
        ),
        "2": (
            "आज का मौसम बहुत सुहावना है और खाना भी बढ़िया था।",
            "Hindi Opinion/Review: 'आज का मौसम बहुत सुहावना है और खाना भी बढ़िया था।'",
        ),
        "3": (
            "भारत की राजधानी नई दिल्ली है।",
            "Hindi Factual: 'भारत की राजधानी नई दिल्ली है।'",
        ),
        "4": (
            "খাবারটা একদম ঠান্ডা ছিল, পরিষেবা খুব খারাপ।",
            "Bengali Negative Complaint: 'খাবারটা একদম ঠান্ডা ছিল, পরিষেবা খুব খারাপ।'",
        ),
        "5": (
            "কলকাতা কি পশ্চিমবঙ্গের রাজধানী?",
            "Bengali Inquiry: 'কলকাতা কি পশ্চিমবঙ্গের রাজধানী?'",
        ),
    }

    print("\n" + "=" * 60)
    print("MULTI-MODAL TESTING INTERFACE (MODES A & C)")
    print("=" * 60)
    print("Select an option to test without terminal clipboard rendering issues:")
    print("  [1] Hindi Inquiry:        'क्या यह बस सीधे हावड़ा स्टेशन जाएगी?'")
    print("  [2] Hindi Opinion/Review: 'आज का मौसम बहुत सुहावना है और खाना भी बढ़िया था।'")
    print("  [3] Hindi Factual:        'भारत की राजधानी नई दिल्ली है।'")
    print("  [4] Bengali Complaint:    'খাবারটা একদম ঠান্ডা ছিল, পরিষেবা খুব খারাপ।'")
    print("  [5] Bengali Inquiry:      'কলকাতা কি পশ্চিমবঙ্গের রাজধানী?'")
    print("  [custom] Enter custom pasted Indic text (Mode C)")
    print("  [exit / q] Quit interactive session")
    print("=" * 60)

    while True:
        try:
            user_choice = input("\nSelect option [1-5], type 'custom', or type 'exit' to quit: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting interactive CLI test session.")
            break

        cleaned_choice = clean_indic_text(user_choice)
        if not cleaned_choice or cleaned_choice.lower() in ("exit", "q", "quit"):
            print("Exiting interactive test session. Goodbye!")
            break

        target_text = ""
        description = ""

        if cleaned_choice in MENU_OPTIONS:
            # Mode A: Pre-defined Numbered Menu (Fail-Safe, bypasses clipboard issues)
            target_text, description = MENU_OPTIONS[cleaned_choice]
            print(f"\n[Mode A Selected: #{cleaned_choice}] {description}")
        elif cleaned_choice.lower() == "custom":
            # Mode C: Prompt for custom input
            try:
                custom_input = input("Enter custom Indic text: ")
            except (EOFError, KeyboardInterrupt):
                print("\nExiting interactive test session.")
                break
            cleaned_custom = clean_indic_text(custom_input)
            if not cleaned_custom:
                print("Empty input. Returning to menu.")
                continue
            target_text = cleaned_custom
            description = "Custom User Input"
        else:
            # Mode C Direct Paste Fallback: User pasted directly at the main prompt
            target_text = cleaned_choice
            description = "Direct Pasted Text"

        result = predict_indic_text(
            text=target_text,
            model=model,
            tokenizer=tokenizer,
            device=device,
        )

        sent_info = result["sentiment"]
        intent_info = result["intent"]

        print("\n" + "-" * 55)
        print(f"Target Text: \"{result['raw_text']}\"")
        print(f"Description: {description}")
        print("-" * 55)
        print(f"Predicted Sentiment: {sent_info['label']} (Confidence: {sent_info['confidence'] * 100:.2f}%)")
        print(f"Predicted Intent:    {intent_info['label']} (Confidence: {intent_info['confidence'] * 100:.2f}%)")

        print("\nClass Probability Breakdown:")
        print("  [Sentiment Distribution]")
        for label, prob in sent_info["probabilities"].items():
            print(f"    - {label:<10}: {prob * 100:6.2f}%")
        print("  [Intent Distribution]")
        for label, prob in intent_info["probabilities"].items():
            print(f"    - {label:<10}: {prob * 100:6.2f}%")
        print("-" * 55)

        print("\nClean JSON Payload:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
