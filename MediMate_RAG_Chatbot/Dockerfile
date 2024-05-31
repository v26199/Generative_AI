# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Load environment variables from .env file
RUN apt-get update && apt-get install -y dos2unix
COPY .env ./
RUN dos2unix .env   


# Run app.py when the container launches
CMD ["python", "app.py"]
