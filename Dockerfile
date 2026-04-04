FROM python:3.11-slim

WORKDIR /app

# Enable the Gradio web interface for testing
ENV ENABLE_WEB_INTERFACE=true

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Install the package itself so absolute imports work
RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
