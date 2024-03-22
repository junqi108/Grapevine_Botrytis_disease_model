ARG PYTHON_TAG

FROM python:${PYTHON_TAG} as base

WORKDIR /workspace

ENV PYTHONDONTWRITEBYTECODE 1

ENV PYTHONUNBUFFERED 1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libpq-dev graphviz  \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/downloaded_packages

FROM base as build

COPY requirements.txt ./

RUN pip wheel --no-cache-dir --no-deps --wheel-dir /workspace/wheels -r requirements.txt

FROM base as runtime

COPY --from=build /workspace/wheels /wheels

COPY --from=build /workspace/requirements.txt .

RUN pip install --no-cache /wheels/*
