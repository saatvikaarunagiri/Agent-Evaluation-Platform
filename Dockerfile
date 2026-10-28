# Image for the agent-eval Python services (ingest worker, training).
# Default REQS is the light CI set; pass the full requirements for the
# ModernBERT + DPO/GRPO paths.
#
#   docker build -t agent-eval .
#   docker build -t agent-eval-full --build-arg REQS=p1_agent_eval/requirements.txt .

FROM python:3.12-slim

ARG REQS=requirements-ci.txt
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-ci.txt ./requirements-ci.txt
COPY p1_agent_eval/requirements.txt ./p1_agent_eval/requirements.txt
RUN pip install --upgrade pip && pip install -r ${REQS}

COPY . .

CMD ["python", "-m", "p1_agent_eval.demo"]
