<img width="1186" height="1256" alt="text_summarizer" src="https://github.com/user-attachments/assets/2c541b71-957b-434f-bdb8-5bdb99936212" />

# text-summarizer

**Fine-tune a transformer to summarize conversations.** This project takes Google's Pegasus model and fine-tunes it on the **SAMSum** dialogue dataset, so it can turn a multi-turn chat into a short summary — wrapped in a clean, config-driven pipeline, served through a Streamlit UI over FastAPI, and deployed serverless on Azure.

> **Live demo:** https://text-summarizer.grayglacier-3d9dbaf5.koreacentral.azurecontainerapps.io
> It runs serverless (scale-to-zero), so the **first** request after it's been idle spins the container up and loads the model — give it up to a minute. After that it's quick.

---

## Why I built this

I wanted a project that demonstrated the **full model lifecycle**, not just calling an API: taking a pretrained transformer and actually *fine-tuning* it on a new task, with the whole thing structured the way a real ML codebase is — config-driven components, typed configuration objects, staged pipelines, shared logging/error-handling — rather than one long notebook.

The base model, `google/pegasus-cnn_dailymail`, is already good at summarizing *news articles*. The interesting part is adapting it to a different shape of text — informal, multi-speaker **dialogue** — by fine-tuning on SAMSum, then measuring whether that adaptation actually worked with ROUGE.

---

## What it does

- **Fine-tunes Pegasus on SAMSum** through a five-stage pipeline, each stage a self-contained, config-driven component.
- **Evaluates with ROUGE** against the test set, so the result is a number, not a vibe.
- **Serves a Streamlit UI** (paste a conversation, get a summary) backed by a **FastAPI** `/predict` endpoint.
- **Runs deployed** on Azure Container Apps as a single Docker image — serverless, scaling to zero when idle.

The five training stages:

1. **Data ingestion** — downloads and unzips the SAMSum dataset.
2. **Data validation** — confirms the expected `train` / `test` / `validation` splits are present.
3. **Data transformation** — tokenizes each dialogue/summary pair with the Pegasus tokenizer and saves the encoded dataset.
4. **Model trainer** — fine-tunes Pegasus with the HuggingFace `Trainer`.
5. **Model evaluation** — generates summaries on the test set and scores them with ROUGE.

---

## Architecture

```
TRAINING (run once, needs a GPU)
config.yaml + params.yaml
        |   read via ConfigurationManager -> typed dataclass entities
        v
 +-----------+   +------------+   +----------------+   +-------------+   +-----------------+
 | Ingestion |-->| Validation |-->| Transformation |-->| Model       |-->| Model           |
 | get SAMSum|   | check files|   | tokenize       |   | Trainer     |   | Evaluation      |
 +-----------+   +------------+   +----------------+   | fine-tune   |   | ROUGE on test   |
                                                       +------+------+   +-----------------+
                                                              | saves model + tokenizer
                                                              v
                                                   artifacts/model_trainer/

SERVING (one container, two servers)
   browser --> Streamlit UI (public :8080) --HTTP--> FastAPI /predict (internal :8000)
                                                          |
                                                          v
                                              fine-tuned Pegasus (loaded once)

   packaged as a Docker image -> Azure Container Apps (serverless, scale-to-zero)
```

Every stage reads its settings through a single `ConfigurationManager`, which turns the YAML into frozen `@dataclass` config objects — so no component ever touches a YAML file directly.

**Stack:** Python 3.11 · HuggingFace `transformers` + `datasets` + `evaluate` · PyTorch · Pegasus (`google/pegasus-cnn_dailymail`) · FastAPI · Streamlit · Docker · Azure Container Apps

---

## Design decisions

A few deliberate choices, including the tradeoffs — those are the honest part.

**Config-driven components with typed entities.** Each stage's settings live in `config.yaml` / `params.yaml`, get read once by a `ConfigurationManager`, and are handed to components as frozen dataclasses. More upfront structure than a script, but every path/hyperparameter is traceable to one place and each stage is testable in isolation.

**Pegasus fine-tuned, not trained from scratch.** Starting from a model already strong at summarization means the fine-tune only teaches it the *dialogue* domain, which is realistic on a single consumer GPU. The cost: the base model's news habits (e.g. its `<n>` newline token) leak through and get cleaned up at decode time.

**Demo-scale training, on purpose.** One epoch, `fp16`, batch size 1 with gradient accumulation — tuned to *run* on one GPU in reasonable time, not to top a leaderboard. The ROUGE number is a sanity check that fine-tuning moved the needle, not a SOTA claim.

**Model loaded once, not per request.** The FastAPI serving layer builds the summarization pipeline a single time and reuses it, so only a genuine cold start pays the ~2.2 GB load — warm requests are fast.

**Serverless, scale-to-zero deployment.** The container runs on Azure Container Apps (Consumption plan, 2 vCPU / 4 GiB). It scales to zero when idle, so it costs nothing between visits; the tradeoff is a cold start (container spin-up + model load) on the first request after idle — an accepted trade for a portfolio demo with bursty traffic. Inference runs on **CPU** (no GPU on this tier), so summaries take a few seconds.

**`/train` runs the pipeline as a subprocess.** The `/train` route shells out to run `main.py` — a deliberate simplification for a demo; a production system would hand training to a task queue. Documented, not pretended-away.

---

## Evaluation

ROUGE for the fine-tuned Pegasus model, scored on the first **100** dialogues of the SAMSum test split.

| Metric | Score |
|---|---|
| ROUGE-1 | 28.37 |
| ROUGE-2 | 8.22 |
| ROUGE-L | 21.89 |
| ROUGE-Lsum | 21.93 |

_(F-scores ×100. Reproduce with `python main.py` through the evaluation stage; raw values land in `artifacts/model_evaluation/metrics.csv`.)_

---

## Running it locally

Training and evaluation need a **CUDA GPU**. Inference runs on CPU (as it does in the deployed container), just slower.

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

**3. Train** (ingestion -> validation -> transformation -> training -> evaluation):

```bash
python main.py
```

This populates `artifacts/` with the dataset, the fine-tuned model, and `metrics.csv`.

**4. Serve** — either run the app directly, or the way it's deployed (via Docker):

```bash
python app.py          # FastAPI only, on http://localhost:8000
# or, the full deployed setup (Streamlit UI + FastAPI):
docker build -t textsummarizer .
docker run -p 8080:8080 textsummarizer   # open http://localhost:8080
```

---

## Deployment

The app is packaged as a single Docker image (`parammalviya/textsummarizer` on Docker Hub) and deployed to **Azure Container Apps** on the serverless Consumption plan.

- **One container, two servers:** a `start.sh` launches FastAPI (internal port 8000) and Streamlit (public port 8080); Streamlit calls the API over localhost. The image installs the **CPU** build of PyTorch, since the platform has no GPU.
- **Slim image:** the `.dockerignore` bakes in only the final model + tokenizer, dropping the HuggingFace training checkpoint (which alone was ~6.5 GB).
- **Scale-to-zero:** `min-replicas` is 0, so idle costs nothing; the first request after idle is a cold start.

---

## Config

Non-secret settings live in `config.yaml` and `params.yaml`:

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
ui/                 streamlit_app.py  (the summarizer UI)
research/           development notebooks
app.py              FastAPI serving layer (/train, /predict)
main.py             runs the full training pipeline
start.sh            launches FastAPI + Streamlit in the container
Dockerfile          builds the serving image
```
