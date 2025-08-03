# Bajaj Finserv Insurance API - Docker Setup

## Build and Run

1. **Build the Docker image:**
   ```bash
   docker build -t bajaj-finserv-insurance-api .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8000:8000 bajaj-finserv-insurance-api
   ```

## Notes
- All code files (`app.py`, `main.py`, etc.) should be in the root directory.
- The `.gitignore` prevents committing cache, environments, and local data.
- Update `requirements.txt` if you add dependencies.
