
# Set the working directory in the container to /app
WORKDIR /app

ADD ./requirements.txt /app
# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN transformers-cli download --model vinvino02/glpn-nyu

# feature_extractor = GLPNImageProcessor.from_pretrained("/app/glpn-nyu")
# model = GLPNForDepthEstimation.from_pretrained("/app/glpn-nyu")


# Add the current directory contents into the container at /app
ADD . /app

# Make port 80 available to the world outside this container
EXPOSE 80

# Run the command to start your application
CMD ["python", "main.py"]

# docker build -t 2dto3dto2d:latest .

# docker run --gpus all -v /host/path/to/input:/app/media 2dto3dto2d:latest