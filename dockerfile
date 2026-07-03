FROM python:3.8-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir fastapi uvicorn pydantic numpy

# Copy project files
COPY Models/ ./Models/
COPY losses/ ./losses/
COPY api.py .

# Expose port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]