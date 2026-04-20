#!/bin/bash
# First-boot script for the ORB VM: Docker only (containers pulled after SSH).
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y docker.io
systemctl enable docker
systemctl start docker
