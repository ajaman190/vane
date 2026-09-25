# CPU image. Weights are not downloaded during build.
FROM python:3.12-slim

WORKDIR /app

RUN groupadd --gid 1000 vane \
    && useradd --uid 1000 --gid vane --create-home --shell /bin/bash vane

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e ".[serve]" \
    && chown -R vane:vane /app

USER vane

EXPOSE 8000

CMD ["vane-serve", "--host", "0.0.0.0", "--port", "8000"]
