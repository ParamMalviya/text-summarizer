# text-summarizer# text-summarizer

**Fine-tune a transformer to summarize conversations.** This project takes Google's Pegasus model and fine-tunes it on the **SAMSum** dialogue dataset, so it can turn a multi-turn chat into a short summary — wrapped in a clean, config-driven pipeline and served over a FastAPI endpoint.

> Not currently deployed — it runs locally (train, then serve). Containerizing and hosting is a planned next step.

---

## Why I built this

I wanted a project that demonstrated the **full model lifecycle**, not just calling an API: taking a pretrained transformer and actually *fine-tuning* it on a new task, with the whole thing structured the way a real ML codebase is — config-driven components, typed configuration objects, staged pipelines, and shared logging/error-handling — rather than one long notebook.

The base model, `google/pegasus-cnn_dailymail`, is already good at summarizing *news articles*. The interesting part is adapting it to a different shape of text — informal, multi-speaker **dialogue** — by fine-tuning on SAMSum, then measuring whether that adaptation actually worked with ROUGE.

---

## What it does

- **Fine-tunes Pegasus on SAMSum** through a five-stage pipeline, each stage a self-contained, config-driven component.
- **Evaluates with ROUGE** against the test split, so the result is a number, not a vibe.
- **Serves over FastAPI** — `POST /predict` summarizes a block of text; `POST /train` kicks off the training pipeline.

The five stages:

1. **Data ingestion** — downloads and unzips the SAMSum dataset.
2. **Data validation** — confirms the expected `train` / `test` / `validation` splits are present.
3. **Data transformation** — tokenizes each dialogue/summary pair with the Pegasus tokenizer and saves the encoded dataset to disk.
4. **Model trainer** — fine-tunes Pegasus with the HuggingFace `Trainer`.
5. **Model evaluation** — generates summaries on the test set and scores them with ROUGE.

---

## Architecture

```
config.yaml + params.yaml
        |   read via ConfigurationManager -> typed dataclass entities
        v
 +------------+   +------------+   +----------------+   +--------------+   +------------------+
 | Ingestion  |-->| Validation |-->| Transformation |-->| Model Trainer|-->| Model Evaluation |
 | get SAMSum |   | check files|   | tokenize       |   | fine-tune    |   | ROUGE on test    |
 +------------+   +------------+   +----------------+   +------+-------+   +------------------+
                                                              | saves fine-tuned model + tokenizer
                                                              v
                                                    +-------------------+
                                                    |  FastAPI (app.py) |
                                                    |  /train   /predict|
                                                    +-------------------+
```

Every stage reads its settings through a single `ConfigurationManager`, which turns the YAML into frozen `@dataclass` config objects — so no component ever touches a YAML file directly, and every path/hyperparameter is declared in one place.

**Stack:** Python 3.11 · HuggingFace `transformers` + `datasets` + `evaluate` · PyTorch · Pegasus (`google/pegasus-cnn_dailymail`) · FastAPI · pandas

---

## Design decisions

A few deliberate choices, including the tradeoffs — those are the honest part.

**Config-driven components with typed entities.** Each stage's settings live in `config.yaml` / `params.yaml`, get read once by a `ConfigurationManager`, and are handed to components as frozen dataclasses. Components never parse YAML themselves. It's more upfront structure than a script, but it makes every path and hyperparameter traceable to one place and each stage testable in isolation.

**Pegasus fine-tuned, not trained from scratch.** Starting from a model already strong at summarization means the fine-tune only has to teach it the *dialogue* domain, which is realistic on a single consumer GPU. The cost: the base model's news-summarization habits (e.g. its `<n>` newline token) leak through and have to be cleaned up at decode time.

**Demo-scale training, on purpose.** One epoch, `fp16`, batch size 1 with gradient accumulation — tuned to *run* on one GPU in a reasonable time, not to top a leaderboard. The point of the project is the end-to-end pipeline; the ROUGE number is a sanity check that fine-tuning moved the needle, not a SOTA claim.

**Evaluated on a test slice.** Evaluation currently scores the first **10** test dialogues — fast feedback while iterating. It's a small, noisy sample and is trivially widened (one line in `model_evaluation.py`); I'm calling it out rather than hiding it.

**Shared logging + exception handling.** The `logger/` and `exception/` modules are identical across my projects (DocTalk, this, and student-performance) — same code for the same job, so the infrastructure reads the same everywhere.

**`/train` runs the pipeline as a subprocess.** The `/train` route shells out to run `main.py`. It's a deliberate simplification for a demo — a production system would hand training to a task queue and return immediately, not block a web request. Documented, not pretended-away.

---

## Evaluation

ROUGE on the SAMSum test slice, for the fine-tuned Pegasus model.

| Metric | Score |
|---|---|
| ROUGE-1 | 28.37 |
| ROUGE-2 | 8.22 |
| ROUGE-L | 21.89 |
| ROUGE-Lsum | 21.93 |

_(Scored on the first 100 test dialogues. Reproduce with `python main.py` through the evaluation stage; numbers land in `artifacts/model_evaluation/metrics.csv`.)_

---

## Running it locally

Training and evaluation need a **CUDA GPU** (Pegasus fine-tuning on CPU is impractical). Inference via `/predict` will run on CPU, just slowly.

**1. Clone and set up the environment:**

```bash
git clone https://github.com/ParamMalviya/text-summarizer.git
cd text-summarizer
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

**2. Install PyTorch separately** (its CUDA wheels come from PyTorch's own index, not PyPI):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**3. Run the full training pipeline** (ingestion → validation → transformation → training → evaluation):

```bash
python main.py
```

This populates `artifacts/` with the dataset, the fine-tuned model, and `metrics.csv`.

**4. Serve the model:**

```bash
python app.py         # FastAPI on http://localhost:8080
```

Open `http://localhost:8080/docs`, then call `POST /predict` with a block of dialogue to get a summary back.

---

## Config

Non-secret settings live in `config.yaml` and `params.yaml`, so they're tunable without touching code:

| Setting | Value |
|---|---|
| Base model | `google/pegasus-cnn_dailymail` |
| Dataset | SAMSum (dialogue summarization) |
| Epochs | 1 |
| Precision | fp16 |
| Batch size / grad-accum | 1 / 16 |
| Generation | beam search (`num_beams=8`, `length_penalty=0.8`, `max_length=128`) |

This project uses **no secrets** — the dataset is a public download and the model comes from the public HuggingFace Hub, so there's no `.env` to configure.

---

## Project structure

```
src/textSummarizer/
├── components/     ingestion, validation, transformation, trainer, evaluation
├── pipelines/      one stage_0N wrapper per component + prediction
├── config/         ConfigurationManager
├── constants/      config/params file paths
├── entity/         frozen dataclass configs
├── utils/          read_yaml, create_directories, get_size
├── logger/         shared logging
└── exception/      shared error handling
config/             config.yaml
research/           development notebooks
app.py              FastAPI serving layer
main.py             runs the full training pipeline
```