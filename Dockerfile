# Use Python 3.10 slim base image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app code
COPY . .

# Expose the port your Flask app runs on
EXPOSE 7000

# Run your Flask app
CMD ["python", "app.py"]
