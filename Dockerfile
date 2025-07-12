# Use a stable Python base image
FROM python:3.12.1

# Install AWS Command Line Interface
RUN apt update -y && apt install awscli -y

# Set the working directory inside the container
WORKDIR /app

# Copy the contents of the project directory to /app
COPY . /app

# Install dependencies from requirements.txt
RUN pip install -r requirements.txt

# Run the application
CMD ["python3", "app.py"]
