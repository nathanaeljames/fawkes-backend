# Start from the official Rasa SDK base image
FROM rasa/rasa-sdk:latest

# Switch to root user temporarily to install system dependencies/Python libraries
# The base image defaults to a non-root user (often 'rasa') for security.
USER root

# Install the pytz library globally
# --no-cache-dir saves space
RUN pip install \
    pytz --no-cache-dir\
    aiohttp

# Switch back to the default non-root user for running the action server
# This is crucial for maintaining the security standards of the Rasa SDK image.
USER 1001