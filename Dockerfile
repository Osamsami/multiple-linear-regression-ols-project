FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by matplotlib/pandas wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Jupyter Lab (for the notebook) is exposed on 8888, Streamlit on 8501.
EXPOSE 8888 8501

# Default: launch Jupyter Lab so the notebook can be explored/run.
# Override the command to run the Streamlit demo instead, e.g.:
#   docker run -p 8501:8501 <image> streamlit run app.py --server.address=0.0.0.0
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
