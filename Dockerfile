# Use the official sglang image
FROM lmsysorg/sglang:v0.4.8.post1-cu126

# Install libgl for opencv support
RUN apt-get update && apt-get install -y libgl1 && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install -U Flask pyngrok --break-system-packages
# Install mineru latest
RUN python3 -m pip install -U 'mineru[core]' --break-system-packages

# Download models and update the configuration file
RUN /bin/bash -c "mineru-models-download -s huggingface -m all"

WORKDIR /home
COPY --chown=root:root  main.py     main.py
# Set the entry point to activate the virtual environment and run the command line tool
# ENTRYPOINT ["/bin/bash", "-c", "export MINERU_MODEL_SOURCE=local && exec \"$@\"", "--"]
ENTRYPOINT ["/bin/python3",  "/home/main.py"]

