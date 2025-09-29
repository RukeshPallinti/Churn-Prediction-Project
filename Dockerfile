# Use official Python 3.10 slim image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first to leverage Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all other project files
COPY . .

# Expose the port Render expects
EXPOSE 10000

# Set Flask environment variables
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=10000

# Start the Flask app
CMD ["flask", "run"]
