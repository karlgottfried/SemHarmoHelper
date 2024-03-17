# Use the official Python image as the base
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file to the working directory
COPY requirements.txt ./

RUN pip install --upgrade pip
RUN pip install scikit-learn
# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files into the working directory
COPY . .

# Expose the port on which the Streamlit application runs
EXPOSE 8501

# Execute the Streamlit application
CMD ["streamlit", "run", "Harmo_Helper.py"]
