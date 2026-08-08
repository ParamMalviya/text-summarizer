# Dockerfile -- lives at the repo root. the recipe used to build this app's container.

# slim official Python matching my 3.11 (slim = smaller image, faster builds).
FROM python:3.11-slim

# run as a non-root user (uid 1000), same convention as DocTalk -- safer, and the
# app can still write logs/ because it owns its own workdir.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# everything below happens inside this folder, which "user" owns.
WORKDIR /home/user/app

# 1) CPU build of torch FIRST. Azure Container Apps has NO GPU, so I must use the
#    CPU wheel, NOT the cu121 GPU wheel my laptop uses -- those live on PyTorch's
#    own index, not PyPI. (DocTalk never needed this: it has no torch at all.)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# 2) bake the fine-tuned model + tokenizer in. copied early, on its own layer, so
#    this heavy ~2.2GB step stays cached when I only change code later.
#    (.dockerignore prunes the big training checkpoint, so only these two get in.)
COPY --chown=user artifacts/model_trainer/ ./artifacts/model_trainer/

# 3) install the rest of my pinned deps. requirements.txt ends with "-e ." which
#    installs my own textSummarizer package, so I copy the code it needs first.
COPY --chown=user requirements.txt pyproject.toml ./
COPY --chown=user src/ ./src/
RUN pip install --no-cache-dir -r requirements.txt

# 4) the app, the UI, the launcher, and the config they read at runtime.
COPY --chown=user config/ ./config/
COPY --chown=user ui/ ./ui/
COPY --chown=user params.yaml app.py main.py start.sh ./

# the public port Streamlit serves on.
EXPOSE 8080

# boot both servers via the launcher (FastAPI on 8000 internal, Streamlit on 8080).
CMD ["bash", "start.sh"]