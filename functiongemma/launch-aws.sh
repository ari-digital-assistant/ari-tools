#!/usr/bin/env bash
#
# Launch an instance, fine-tune FunctionGemma on it, download the
# resulting GGUF, and terminate the instance.
#
# Tries g6.xlarge, falls back to g5.xlarge. Tries spot, falls back
# to on-demand. Creates a default VPC if the region doesn't have one.
#
# Prerequisites:
#   - aws CLI configured (default profile, or GitHub OIDC-assumed role)
#   - Secrets Manager secret `ari-functiongemma/hf-token` in the target region
#   - Instance profile `ari-functiongemma-instance` (IAM is global)
#
# Usage:
#   ./launch-aws.sh

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
KEY_NAME="${AWS_KEY_NAME:-ari-functiongemma}"
SECURITY_GROUP_NAME="${AWS_SG_NAME:-ari-functiongemma-train}"
LOCAL_OUT="${LOCAL_OUT:-./output}"
TOOLS_REPO="${TOOLS_REPO:-https://github.com/ari-digital-assistant/ari-tools.git}"
TOOLS_BRANCH="${TOOLS_BRANCH:-main}"
INSTANCE_PROFILE="${AWS_INSTANCE_PROFILE:-ari-functiongemma-instance}"
HF_SECRET_ID="${HF_SECRET_ID:-ari-functiongemma/hf-token}"

mkdir -p "$LOCAL_OUT"

# ── 1. Ensure default VPC exists ─────────────────────────────────────────

echo "[1/9] Checking for default VPC in $REGION..."
DEFAULT_VPC=$(aws ec2 describe-vpcs --region "$REGION" \
    --filters "Name=is-default,Values=true" \
    --query 'Vpcs[0].VpcId' --output text 2>/dev/null || echo "None")

if [[ "$DEFAULT_VPC" == "None" || -z "$DEFAULT_VPC" ]]; then
    echo "  No default VPC found. Creating one..."
    DEFAULT_VPC=$(aws ec2 create-default-vpc --region "$REGION" \
        --query 'Vpc.VpcId' --output text)
    echo "  Created default VPC: $DEFAULT_VPC"
else
    echo "  Default VPC: $DEFAULT_VPC"
fi

# ── 2. Resolve instance type (g6.xlarge → g5.xlarge fallback) ────────────

echo "[2/9] Resolving instance type..."
INSTANCE_TYPE=""
for candidate in g6.xlarge g5.xlarge; do
    AZ_COUNT=$(aws ec2 describe-instance-type-offerings --region "$REGION" \
        --location-type availability-zone \
        --filters "Name=instance-type,Values=$candidate" \
        --query 'length(InstanceTypeOfferings)' --output text 2>/dev/null || echo "0")
    if [[ "$AZ_COUNT" -gt 0 ]]; then
        INSTANCE_TYPE="$candidate"
        echo "  $candidate available in $AZ_COUNT AZ(s)"
        break
    else
        echo "  $candidate not available in $REGION, trying next..."
    fi
done

if [[ -z "$INSTANCE_TYPE" ]]; then
    echo "ERROR: no suitable GPU instance type available in $REGION" >&2
    exit 1
fi

# ── 3. Resolve the latest Deep Learning Base GPU AMI ─────────────────────

echo "[3/9] Resolving latest Deep Learning Base GPU AMI in $REGION..."
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

# ── 4. Ensure key pair exists ────────────────────────────────────────────

KEY_FILE="$HOME/.ssh/${KEY_NAME}.pem"
if ! aws ec2 describe-key-pairs --region "$REGION" --key-names "$KEY_NAME" >/dev/null 2>&1; then
    echo "[4/9] Creating SSH key pair $KEY_NAME..."
    aws ec2 create-key-pair --region "$REGION" --key-name "$KEY_NAME" \
        --query 'KeyMaterial' --output text > "$KEY_FILE"
    chmod 600 "$KEY_FILE"
    echo "  Key saved to $KEY_FILE"
else
    echo "[4/9] SSH key pair $KEY_NAME exists"
fi

# ── 5. Ensure security group exists (allow SSH from your IP) ─────────────

MY_IP=$(curl -s https://checkip.amazonaws.com | tr -d '[:space:]')
echo "[5/9] Ensuring security group $SECURITY_GROUP_NAME (SSH from $MY_IP/32)..."

SG_ID=$(aws ec2 describe-security-groups --region "$REGION" \
    --filters "Name=group-name,Values=$SECURITY_GROUP_NAME" "Name=vpc-id,Values=$DEFAULT_VPC" \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo "None")

if [[ "$SG_ID" == "None" || -z "$SG_ID" ]]; then
    SG_ID=$(aws ec2 create-security-group --region "$REGION" \
        --group-name "$SECURITY_GROUP_NAME" \
        --description "Ari FunctionGemma training SSH access" \
        --query 'GroupId' --output text)
    echo "  Created SG $SG_ID"
fi

aws ec2 authorize-security-group-ingress --region "$REGION" \
    --group-id "$SG_ID" \
    --protocol tcp --port 22 --cidr "$MY_IP/32" 2>/dev/null || true

echo "  SG: $SG_ID"

# ── 6. Build user-data ───────────────────────────────────────────────────

USER_DATA_FILE=$(mktemp)
cat > "$USER_DATA_FILE" <<EOF
#!/bin/bash
set -e
exec > >(tee /var/log/ari-train.log | logger -t ari-train -s 2>/dev/console) 2>&1

cd /home/ubuntu

echo "Cloning ari-tools..."
sudo -u ubuntu git clone --depth 1 --branch "$TOOLS_BRANCH" "$TOOLS_REPO" ari-tools

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

unset HF_TOKEN

echo "Training complete. Touching done marker."
sudo -u ubuntu touch /home/ubuntu/output/DONE
EOF

USER_DATA_B64=$(base64 -w0 "$USER_DATA_FILE")
rm -f "$USER_DATA_FILE"

# ── 7. Launch: try spot, fall back to on-demand ──────────────────────────

LAUNCH_SPEC="{
    \"ImageId\": \"$AMI_ID\",
    \"InstanceType\": \"$INSTANCE_TYPE\",
    \"KeyName\": \"$KEY_NAME\",
    \"SecurityGroupIds\": [\"$SG_ID\"],
    \"IamInstanceProfile\": {\"Name\": \"$INSTANCE_PROFILE\"},
    \"BlockDeviceMappings\": [{
        \"DeviceName\": \"/dev/sda1\",
        \"Ebs\": {\"VolumeSize\": 100, \"VolumeType\": \"gp3\"}
    }],
    \"UserData\": \"$USER_DATA_B64\"
}"

INSTANCE_ID=""

# Attempt 1: spot
echo "[6/9] Trying spot instance ($INSTANCE_TYPE in $REGION)..."
SPOT_REQUEST_ID=$(aws ec2 request-spot-instances --region "$REGION" \
    --instance-count 1 \
    --type "one-time" \
    --launch-specification "$LAUNCH_SPEC" \
    --query 'SpotInstanceRequests[0].SpotInstanceRequestId' --output text 2>/dev/null || echo "")

if [[ -n "$SPOT_REQUEST_ID" ]]; then
    echo "  Spot request: $SPOT_REQUEST_ID"
    echo "  Waiting up to 3 minutes for fulfilment..."

    SPOT_OK=false
    for i in $(seq 1 12); do
        STATE=$(aws ec2 describe-spot-instance-requests --region "$REGION" \
            --spot-instance-request-ids "$SPOT_REQUEST_ID" \
            --query 'SpotInstanceRequests[0].Status.Code' --output text 2>/dev/null || echo "unknown")

        if [[ "$STATE" == "fulfilled" ]]; then
            INSTANCE_ID=$(aws ec2 describe-spot-instance-requests --region "$REGION" \
                --spot-instance-request-ids "$SPOT_REQUEST_ID" \
                --query 'SpotInstanceRequests[0].InstanceId' --output text)
            SPOT_OK=true
            echo "  Spot fulfilled: $INSTANCE_ID"
            break
        elif [[ "$STATE" == "capacity-not-available" || "$STATE" == "capacity-oversubscribed" || "$STATE" == "price-too-low" ]]; then
            echo "  Spot unavailable ($STATE), cancelling..."
            aws ec2 cancel-spot-instance-requests --region "$REGION" \
                --spot-instance-request-ids "$SPOT_REQUEST_ID" >/dev/null 2>&1 || true
            break
        fi
        sleep 15
    done

    if [[ "$SPOT_OK" == "false" && -n "$SPOT_REQUEST_ID" ]]; then
        aws ec2 cancel-spot-instance-requests --region "$REGION" \
            --spot-instance-request-ids "$SPOT_REQUEST_ID" >/dev/null 2>&1 || true
    fi
fi

# Attempt 2: on-demand
if [[ -z "$INSTANCE_ID" ]]; then
    echo "[6/9] Spot unavailable. Launching on-demand ($INSTANCE_TYPE in $REGION)..."
    UD_TMP=$(mktemp)
    echo "$USER_DATA_B64" | base64 -d > "$UD_TMP"
    INSTANCE_ID=$(aws ec2 run-instances --region "$REGION" \
        --image-id "$AMI_ID" \
        --instance-type "$INSTANCE_TYPE" \
        --key-name "$KEY_NAME" \
        --security-group-ids "$SG_ID" \
        --iam-instance-profile "Name=$INSTANCE_PROFILE" \
        --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
        --user-data "file://$UD_TMP" \
        --query 'Instances[0].InstanceId' --output text)
    rm -f "$UD_TMP"

    if [[ -z "$INSTANCE_ID" || "$INSTANCE_ID" == "None" ]]; then
        echo "ERROR: failed to launch on-demand instance" >&2
        exit 1
    fi
    echo "  On-demand instance: $INSTANCE_ID"
fi

# Tag the instance for visibility
aws ec2 create-tags --region "$REGION" --resources "$INSTANCE_ID" \
    --tags "Key=Name,Value=ari-functiongemma-train" || true

# ── 8. Wait for SSH ──────────────────────────────────────────────────────

echo "[7/9] Waiting for instance to be reachable via SSH..."

aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"

PUBLIC_IP=$(aws ec2 describe-instances --region "$REGION" \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "  Public IP: $PUBLIC_IP"

for i in $(seq 1 30); do
    if ssh -i "$KEY_FILE" -o StrictHostKeyChecking=no -o IdentitiesOnly=yes -o ConnectTimeout=5 \
        ubuntu@"$PUBLIC_IP" "echo ok" >/dev/null 2>&1; then
        echo "  SSH ready"
        break
    fi
    sleep 10
done

# ── 9. Poll for training completion ──────────────────────────────────────

echo "[8/9] Training in progress. Polling for completion (every 60s)..."
echo "  Watch live: ssh -i $KEY_FILE ubuntu@$PUBLIC_IP 'sudo tail -f /var/log/ari-train.log'"

while true; do
    if ssh -i "$KEY_FILE" -o StrictHostKeyChecking=no -o IdentitiesOnly=yes \
        ubuntu@"$PUBLIC_IP" "test -f /home/ubuntu/output/DONE" 2>/dev/null; then
        echo "  Done."
        break
    fi
    sleep 60
done

# ── 10. Download result and terminate ────────────────────────────────────

echo "[9/9] Downloading GGUF and terminating instance..."

scp -i "$KEY_FILE" -o StrictHostKeyChecking=no -o IdentitiesOnly=yes \
    ubuntu@"$PUBLIC_IP":/home/ubuntu/output/ari-functiongemma-q4_k_m.gguf \
    "$LOCAL_OUT/ari-functiongemma-q4_k_m.gguf"

aws ec2 terminate-instances --region "$REGION" --instance-ids "$INSTANCE_ID" >/dev/null
echo "  Instance terminated."

ls -lh "$LOCAL_OUT/ari-functiongemma-q4_k_m.gguf"
echo
echo "Done. Model: $LOCAL_OUT/ari-functiongemma-q4_k_m.gguf"
