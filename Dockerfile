# Use the official Python image as the base
FROM  python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Upgrade pip and install dependencies in one layer to keep the image clean and small
COPY requirements.txt ./

#RUN apt-get update && apt-get install -y --no-install-recommends \
#    build-essential \
#    curl \
#    software-properties-common \
#    git \
#    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application files into the working directory
COPY . .

# Expose the port on which the Streamlit application runs
EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "Harmo_Helper.py", "--server.port=8501"]
# , "--server.address=192.0.0.0"]

