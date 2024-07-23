#!/bin/bash

# Variables - replace these with your actual values
DOCKER_USERNAME="amirosik"
REPOSITORY_NAME="tm5mlops"
IMAGE_TAG="latest"

# Step 1: Build the Docker image
echo "Building Docker image..."
docker build -t ${DOCKER_USERNAME}/${REPOSITORY_NAME} ../api/Dockerfile

# Step 2: Push the Docker image to Docker Hub
echo "Pushing Docker image to Docker Hub..."
#docker login # you already should be log in
docker push ${DOCKER_USERNAME}/${REPOSITORY_NAME}

# Step 3: Deploy the Docker container
echo "Deploying Docker container..."
docker run -d --name ${REPOSITORY_NAME}_container -p 5153:8080 ${DOCKER_USERNAME}/${REPOSITORY_NAME}:${IMAGE_TAG}

# Step 4: Verify deployment
echo "Verifying deployment..."
docker ps -a
