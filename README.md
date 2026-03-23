# Agentic-Modeling-Operational-Engineering-Beta-Application
Project of multi-agent system for Kaggle, realized in course of "Agentic Systems" by MWS

## Installation
1. Fill in .env file
2. Configure docker nvidia support
2. https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#with-apt-ubuntu-debian
3. https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker (с docker desktop не получится)
4. set -a; source .env; set +a; chmod u+x ./deploy/vllm/coder_start.sh; . ./deploy/vllm/coder_start.sh;
