FROM python:3.11-slim

# Set up user for Hugging Face Spaces (ID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH

# Set working directory
WORKDIR $HOME/app

# Install system dependencies for ChromaDB and Streamlit
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create directories with correct permissions
RUN mkdir -p $HOME/.streamlit
RUN chown user:user $HOME/.streamlit
RUN mkdir -p $HOME/app/data/chroma
RUN chown user:user $HOME/app/data/chroma

# Copy requirements.txt and install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy app directory
COPY --chown=user app/ ./app/

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1

# Run Streamlit
ENTRYPOINT ["streamlit", "run", "app/main.py", "--server.port=7860", "--server.address=0.0.0.0"]