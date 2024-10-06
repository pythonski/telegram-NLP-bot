# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the working directory contents into the container at /usr/src/app
COPY . .

# Set environment variable for Google Cloud credentials
ENV GOOGLE_APPLICATION_CREDENTIALS=google.json

# Run the application
CMD ["python", "main.py"]
