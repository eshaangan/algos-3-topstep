# GCP Deployment Guide

This directory contains scripts to deploy the trading bot to Google Cloud Platform (GCP) using a Compute Engine VM running a Docker container.

## Prerequisites

1.  **GCP Account**: You need an active GCP project.
2.  **gcloud CLI**: Installed and authenticated (`gcloud auth login`, `gcloud config set project YOUR_PROJECT_ID`).
3.  **Docker**: Installed locally.
4.  **APIs Enabled**: Compute Engine API and Container Registry API.

## Setup

1.  **Configure `.env`**: Ensure you have a `.env` file in the project root with your TopstepX credentials:
    ```
    TOPSTEPX_USERNAME=...
    TOPSTEPX_PROJECTX_API_KEY=...
    TOPSTEPX_ACCOUNT_ID=...
    TOPSTEPX_CONTRACT_ID=...
    ```

2.  **Make script executable**:
    ```bash
    chmod +x gcp_deploy/deploy.sh
    ```

## Deployment

Run the deployment script from the **project root**:

```bash
./gcp_deploy/deploy.sh
```

### What it does:
1.  Builds a Docker image containing your code and model.
2.  Pushes the image to Google Container Registry (`gcr.io`).
3.  Creates (or updates) a Compute Engine VM (`topstep-trader-vm`).
4.  Injects your `.env` variables into the container environment.

## Monitoring

-   **View Logs**:
    ```bash
    gcloud compute instances get-serial-port-output topstep-trader-vm --zone=us-central1-a
    ```
    Or better, go to the **GCP Console > Compute Engine**, click the VM, and view **Cloud Logging**.

-   **SSH into VM**:
    ```bash
    gcloud compute ssh topstep-trader-vm --zone=us-central1-a
    ```

-   **Stop the Bot**:
    ```bash
    gcloud compute instances stop topstep-trader-vm --zone=us-central1-a
    ```

## Updating

To deploy code changes or a new model, just run `./gcp_deploy/deploy.sh` again. It will rebuild the image and update the running VM.
