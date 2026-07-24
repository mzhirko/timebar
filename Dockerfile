# Timebar — chronology tool. Build:  docker build -t timebar .
# Fully offline usage (pre-extracted TDGs):
#   docker run --rm -v "$PWD/mycase:/case" timebar \
#       build /case/tdgs -o /case/out --from-tdgs
# Viewer:
#   docker run --rm -p 8501:8501 -v "$PWD/mycase:/case" timebar view /case
# LLM extraction against local Ollama: see docker-compose.yml.
FROM python:3.12-slim

WORKDIR /app
COPY packages/ packages/
COPY rulepacks/ rulepacks/
COPY examples/ examples/
COPY README.md DISCLAIMER.md DATA_POLICY.md LICENSE ./

RUN pip install --no-cache-dir ./packages/tdg-core \
    && pip install --no-cache-dir "./packages/tdg-chrono[llm,pdf,viewer]"

# The viewer binds all interfaces inside the container
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
EXPOSE 8501

ENTRYPOINT ["tdg-chrono"]
CMD ["--help"]
