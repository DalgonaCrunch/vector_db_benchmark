.PHONY: install start-opensearch stop-opensearch run app clean

# Install Python dependencies via uv
install:
	uv sync

# Download and start OpenSearch in background (no Docker required)
start-opensearch:
	bash scripts/start_opensearch.sh --daemon

# Stop the background OpenSearch process
stop-opensearch:
	bash scripts/stop_opensearch.sh

# Run the full benchmark (Qdrant runs in-process by default)
run:
	uv run python main.py

# Remove downloaded OpenSearch binary and generated outputs
clean:
	rm -rf .opensearch/ qdrant_data/ benchmark_results.csv __pycache__

# Launch the Streamlit comparison UI (requires data ingested via `make run`)
app:
	uv run streamlit run app.py --server.headless true
