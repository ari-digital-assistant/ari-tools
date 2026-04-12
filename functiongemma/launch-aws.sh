#!/usr/bin/env bash
#
# Launch a spot instance, fine-tune FunctionGemma on it, download the
# resulting GGUF, and terminate the instance.
#
# Prerequisites:
#   - aws CLI configured (default profile, or GitHub OIDC-assumed role)
#   - Secrets Manager secret `ari-functiongemma/hf-token` holding the
#     HuggingFace token (created once out of band)
#   - Instance profile `ari-functiongemma-instance` (created once) that
#     the spot instance assumes to read that secret
#
# Usage:
#   ./launch-aws.sh
#
# The instance clones ari-tools from GitHub, reads the HF token from
# Secrets Manager, regenerates the dataset fresh from the current Ari
# skill descriptions (so any skill changes you've merged are reflected),
# trains, and the script SCPs the GGUF back here.
#
# Defaults:
#   Region:   eu-west-2 (London)
#   Instance: g6.xlarge (L4, 24GB VRAM, ~$0.85/hr on-demand, ~$0.30/hr spot)
#   AMI:      Deep Learning Base GPU AMI (Ubuntu 22.04, PyTorch + CUDA)

set -euo pipefail

# Disable AWS CLI pager so commands don't block waiting for 'q'.
export AWS_PAGER=""

# Load .env from the repo root if present.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.env"
    set +a
fi

REGION="${AWS_REGION:-eu-west-2}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g6.xlarge}"
KEY_NAME="${AWS_KEY_NAME:-ari-functiongemma}"
SECURITY_GROUP_NAME="${AWS_SG_NAME:-ari-functiongemma-train}"
LOCAL_OUT="${LOCAL_OUT:-./output}"
TOOLS_REPO="${TOOLS_REPO:-https://github.com/ari-digital-assistant/ari-tools.git}"
TOOLS_BRANCH="${TOOLS_BRANCH:-main}"
INSTANCE_PROFILE="${AWS_INSTANCE_PROFILE:-ari-functiongemma-instance}"
HF_SECRET_ID="${HF_SECRET_ID:-ari-functiongemma/hf-token}"

mkdir -p "$LOCAL_OUT"

# ── Check G instance vCPU quota ──────────────────────────────────────────

echo "[0/8] Checking AWS service quota for G instances in $REGION..."
# L-DB2E81BA = "Running On-Demand G and VT instances" (vCPU count, applies to spot too)
QUOTA=$(aws service-quotas get-service-quota --region "$REGION" \
    --service-code ec2 --quota-code L-DB2E81BA \
    --query 'Quota.Value' --output text 2>/dev/null || echo "0")

if [[ "$QUOTA" == "0.0" || "$QUOTA" == "0" ]]; then
    echo "ERROR: G instance vCPU quota is 0 in $REGION." >&2
    echo "Request a quota increase here:" >&2
    echo "  https://${REGION}.console.aws.amazon.com/servicequotas/home/services/ec2/quotas/L-DB2E81BA" >&2
    echo "Ask for at least 4 vCPUs (g6.xlarge needs 4)." >&2
    exit 1
fi
echo "  G instance vCPU quota: $QUOTA"

# ── Resolve the latest Deep Learning Base GPU AMI for Ubuntu 22.04 ────────

echo "[1/8] Resolving latest Deep Learning Base GPU AMI in $REGION..."
AMI_ID=$(aws ec2 describe-images --region "$REGION" \
    --owners amazon \
    --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
              "Name=state,Values=available" \
    --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
    --output text)

if [[ -z "$AMI_ID" || "$AMI_ID" == "None" ]]; then
    echo "ERROR: could not find Deep Learning AMI in $REGION" >&2
    exit 1
fi
echo "  AMI: $AMI_ID"

# ── Ensure key pair exists ───────────────────────────────────────────────

KEY_FILE="$HOME/.ssh/${KEY_NAME}.pem"
if ! aws ec2 describe-key-pairs --region "$REGION" --key-names "$KEY_NAME" >/dev/null 2>&1; then
    echo "[2/8] Creating SSH key pair $KEY_NAME..."
    aws ec2 create-key-pair --region "$REGION" --key-name "$KEY_NAME" \
        --query 'KeyMaterial' --output text > "$KEY_FILE"
    chmod 600 "$KEY_FILE"
    echo "  Key saved to $KEY_FILE"
else
    echo "[2/8] SSH key pair $KEY_NAME exists"
fi

# ── Ensure security group exists (allow SSH from your IP) ────────────────

MY_IP=$(curl -s https://checkip.amazonaws.com | tr -d '[:space:]')
echo "[3/8] Ensuring security group $SECURITY_GROUP_NAME (SSH from $MY_IP/32)..."

SG_ID=$(aws ec2 describe-security-groups --region "$REGION" \
    --filters "Name=group-name,Values=$SECURITY_GROUP_NAME" \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo "None")

if [[ "$SG_ID" == "None" || -z "$SG_ID" ]]; then
    SG_ID=$(aws ec2 create-security-group --region "$REGION" \
        --group-name "$SECURITY_GROUP_NAME" \
        --description "Ari FunctionGemma training SSH access" \
        --query 'GroupId' --output text)
    echo "  Created SG $SG_ID"
fi

# Idempotent: add rule if not present
aws ec2 authorize-security-group-ingress --region "$REGION" \
    --group-id "$SG_ID" \
    --protocol tcp --port 22 --cidr "$MY_IP/32" 2>/dev/null || true

echo "  SG: $SG_ID"

# ── Build user-data: bootstrap script that runs the training ─────────────

USER_DATA_FILE=$(mktemp)
cat > "$USER_DATA_FILE" <<EOF
#!/bin/bash
set -e
exec > >(tee /var/log/ari-train.log | logger -t ari-train -s 2>/dev/console) 2>&1

cd /home/ubuntu

# The Deep Learning AMI has Python + CUDA + awscli. Clone our scripts.
echo "Cloning ari-tools..."
sudo -u ubuntu git clone --depth 1 --branch "$TOOLS_BRANCH" "$TOOLS_REPO" ari-tools

# Fetch the HF token from Secrets Manager using the instance profile's
# auto-rotated credentials. The token never touches user-data or any
# long-lived storage on the instance.
echo "Fetching HF token from Secrets Manager..."
HF_TOKEN=\$(aws secretsmanager get-secret-value \\
    --region "$REGION" \\
    --secret-id "$HF_SECRET_ID" \\
    --query SecretString --output text)

if [[ -z "\$HF_TOKEN" ]]; then
    echo "ERROR: failed to fetch HF token from Secrets Manager" >&2
    exit 1
fi

echo "Running training..."
cd /home/ubuntu/ari-tools/functiongemma
sudo -u ubuntu HF_TOKEN="\$HF_TOKEN" python3 train.py \\
    --hf-token "\$HF_TOKEN" \\
    --output-dir /home/ubuntu/output

# Scrub the token from the environment before marking done.
unset HF_TOKEN

echo "Training complete. Touching done marker."
sudo -u ubuntu touch /home/ubuntu/output/DONE

# Don't auto-shutdown — the launch script polls for DONE and SCPs the result.
EOF

# ── Request spot instance ────────────────────────────────────────────────

echo "[4/8] Requesting spot instance ($INSTANCE_TYPE)..."

SPOT_REQUEST_ID=$(aws ec2 request-spot-instances --region "$REGION" \
    --instance-count 1 \
    --type "one-time" \
    --launch-specification "{
        \"ImageId\": \"$AMI_ID\",
        \"InstanceType\": \"$INSTANCE_TYPE\",
        \"KeyName\": \"$KEY_NAME\",
        \"SecurityGroupIds\": [\"$SG_ID\"],
        \"IamInstanceProfile\": {\"Name\": \"$INSTANCE_PROFILE\"},
        \"BlockDeviceMappings\": [{
            \"DeviceName\": \"/dev/sda1\",
            \"Ebs\": {\"VolumeSize\": 100, \"VolumeType\": \"gp3\"}
        }],
        \"UserData\": \"$(base64 -w0 "$USER_DATA_FILE")\"
    }" \
    --query 'SpotInstanceRequests[0].SpotInstanceRequestId' --output text)

echo "  Spot request: $SPOT_REQUEST_ID"
rm -f "$USER_DATA_FILE"

# ── Wait for fulfilment ──────────────────────────────────────────────────

echo "[5/8] Waiting for spot fulfilment (up to 10 minutes)..."

# Use the built-in waiter — it polls every 15s for up to 40 attempts (10 min).
if ! aws ec2 wait spot-instance-request-fulfilled --region "$REGION" \
    --spot-instance-request-ids "$SPOT_REQUEST_ID" 2>/dev/null; then
    echo "ERROR: spot request not fulfilled within 10 minutes" >&2
    aws ec2 describe-spot-instance-requests --region "$REGION" \
        --spot-instance-request-ids "$SPOT_REQUEST_ID" \
        --query 'SpotInstanceRequests[0].Status' --output json >&2
    exit 1
fi

INSTANCE_ID=$(aws ec2 describe-spot-instance-requests --region "$REGION" \
    --spot-instance-request-ids "$SPOT_REQUEST_ID" \
    --query 'SpotInstanceRequests[0].InstanceId' --output text)
echo "  Instance: $INSTANCE_ID"

# Tag the instance for visibility
aws ec2 create-tags --region "$REGION" --resources "$INSTANCE_ID" \
    --tags "Key=Name,Value=ari-functiongemma-train" || true

# ── Wait for SSH ─────────────────────────────────────────────────────────

echo "[6/8] Waiting for instance to be reachable via SSH..."

aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"

PUBLIC_IP=$(aws ec2 describe-instances --region "$REGION" \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "  Public IP: $PUBLIC_IP"

for i in $(seq 1 30); do
    if ssh -i "$KEY_FILE" -o StrictHostKeyChecking=no -o ConnectTimeout=5 \
        ubuntu@"$PUBLIC_IP" "echo ok" >/dev/null 2>&1; then
        echo "  SSH ready"
        break
    fi
    sleep 10
done

# ── Poll for training completion ─────────────────────────────────────────

echo "[7/8] Training in progress. Polling for completion (every 60s)..."
echo "  Watch live: ssh -i $KEY_FILE ubuntu@$PUBLIC_IP 'sudo tail -f /var/log/ari-train.log'"

while true; do
    if ssh -i "$KEY_FILE" -o StrictHostKeyChecking=no \
        ubuntu@"$PUBLIC_IP" "test -f /home/ubuntu/output/DONE" 2>/dev/null; then
        echo "  Done."
        break
    fi
    sleep 60
done

# ── Download result and terminate ────────────────────────────────────────

echo "[8/8] Downloading GGUF and terminating instance..."

scp -i "$KEY_FILE" -o StrictHostKeyChecking=no \
    ubuntu@"$PUBLIC_IP":/home/ubuntu/output/ari-functiongemma-q4_k_m.gguf \
    "$LOCAL_OUT/ari-functiongemma-q4_k_m.gguf"

aws ec2 terminate-instances --region "$REGION" --instance-ids "$INSTANCE_ID" >/dev/null
echo "  Instance terminated."

ls -lh "$LOCAL_OUT/ari-functiongemma-q4_k_m.gguf"
echo
echo "Done. Model: $LOCAL_OUT/ari-functiongemma-q4_k_m.gguf"
