# LCR-PTM: Fine-Tuning Generative Protein Language Models with Token-Level Label Collision Resolution

LCR-PTM provides training and inference pipelines for adapting generative protein language models (PLMs) to residue-level protein sequence classification tasks. This approaches  include  both **full fine-tuning (Full-FT)** and **parameter-efficient fine-tuning using LoRA (LoRA-FT)** while addressing token-level label collision issues during training.

Supported models include:
* **[ProtGPT2](https://huggingface.co/nferruz/ProtGPT2)**
* **[ProGen-Small](https://huggingface.co/hugohrban/progen2-small)**
* **[ProGen-Medium](https://huggingface.co/hugohrban/progen2-medium)**
---

## Repository Structure

```text
LCR-PTM/
├── bash_scripts/        # Shell scripts for training and inference
├── plm_helpers/         # Data collation, tokenization, metrics, and trainer utilities
├── src/                 # Main training and inference scripts
├── training_logs/       # Training logs (generated during experiments)
├── training_output/     # Model checkpoints and prediction outputs
├── utils/               # Configuration files and evaluation utilities
└── requirements.txt     # Python dependencies
```

---

## Installation

### 1. Create a Virtual Environment

```bash
python -m venv lcrptmvenv
```

### 2. Activate the Environment

**Linux/macOS**

```bash
source lcrptmvenv/bin/activate
```

**Windows**

```bash
lcrptmvenv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the Repository Root to `PYTHONPATH`

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
```

---

## Configuration

Experiment settings are defined in:

```text
utils/configs.json
```

Important configuration options include:


* ProGen model size (`small` or `medium`)
* Tokenization mode
* Rank
* Alpha

---

##  Dataset
The dataset should be placed in `dataset` folder in the format: <br> 
`dataset/{ptm}/train.csv` <br> 
`dataset/{ptm}/test.csv` <br> 
<br> 
For example for Acetylation we have: <br> 
`dataset/acet_k/train.csv` <br> 
`dataset/acet_k/test.csv` <br> 
Columns in the dataset are Seq, Label, UniProtID, pos, full_sequence. Seq has 51  length input window with site of interest centered in the window. Label column has value `0` and `1`. 0 means absence of PTM and 1 means presence of PTM.
## Training

### ProtGPT2

#### LoRA Fine-Tuning

```bash
bash bash_scripts/protgpt2_lora.sh
```

#### Full Fine-Tuning

```bash
bash bash_scripts/protgpt2_full_ft.sh
```

---

### ProGen

LCR-PTM supports two ProGen variants:

* `small`
* `medium`

Specify the desired model size in `utils/configs.json`.

#### LoRA Fine-Tuning

```bash
bash bash_scripts/progen_lora.sh
```

#### Full Fine-Tuning

```bash
bash bash_scripts/progen_full_ft.sh
```

---

## Tokenization Modes

Three tokenization strategies are available and can be selected in `utils/configs.json`.

### 1. `center_split`

Inserts residue boundary markers around the target residue:

```text
<PTM_RESIDUE_START> X <PTM_RESIDUE_END>
```

where `X` denotes the residue of interest.

### 2. `bpe_token`

Inserts motif boundary markers around the local motif containing the target residue:

```text
<PTM_MOTIF_START> motif <PTM_MOTIF_END>
```

### 3. `none`

No special marker tokens are added to the sequence.

---

## Outputs

Training automatically generates:

* Model checkpoints
* Training logs
* Validation metrics
* Test predictions

Outputs are stored in:

```text
training_output/
training_logs/
```

## Running inference

For running inference, the training config(rank, alpha, motif mode and training epochs should be exactly same as training configs in `utils/configs.json` as file names are saved as a combination of config values.)
### ProtGPT2

#### LoRA-finetuned model's inference

```bash
bash src/protgpt2/lora_inference.py
```

#### Full-finetuned model's inference

```bash
bash src/protgpt2/full_ft_inference.py
```


### Progen

#### LoRA-finetuned model's inference

```bash
bash src/progen/lora_inference.py
```

#### Full-finetuned model's inference

```bash
bash src/progen/full_ft_inference.py
```
