$ErrorActionPreference = "Stop"

$ID = "i-04af4feb04601f235"
$PEM = "$env:USERPROFILE\.ssh\face-attendance.pem"

Write-Host "=========================================================="
Write-Host "  STARTING EC2 INSTANCE $ID FOR 80%+ ENSEMBLE RUN"
Write-Host "=========================================================="

aws ec2 start-instances --instance-ids $ID | Out-Null
Write-Host "Waiting for instance to reach running state..."
aws ec2 wait instance-running --instance-ids $ID

$IP = aws ec2 describe-instances --instance-ids $ID `
    --query "Reservations[0].Instances[0].PublicIpAddress" --output text

Write-Host "Public IP: $IP"
Write-Host "Waiting 35 seconds for SSH to be ready..."
Start-Sleep -Seconds 35

$SSH = "ssh -i `"$PEM`" -o StrictHostKeyChecking=accept-new ubuntu@$IP"

Write-Host "Uploading train_pareidolia_ensemble_80plus.py to EC2..."
scp -i "$PEM" -o StrictHostKeyChecking=accept-new "train_pareidolia_ensemble_80plus.py" ubuntu@${IP}:~

Write-Host "Executing 2-Seed 80%+ Ensemble Training Pipeline on GPU..."
$cmd = @"
source activate pytorch 2>/dev/null || source /opt/pytorch/bin/activate 2>/dev/null || true
export LD_LIBRARY_PATH="/home/ubuntu/.local/lib/python3.10/site-packages/nvidia/cudnn/lib:`$LD_LIBRARY_PATH"
python3 train_pareidolia_ensemble_80plus.py
"@
$cmd = $cmd -replace "`r", ""
Invoke-Expression "$SSH '$cmd'"

Write-Host "Downloading lunar_ensemble_80plus_output/ ..."
scp -i "$PEM" -r ubuntu@${IP}:~/lunar_ensemble_80plus_output .

Write-Host "Stopping EC2 instance $ID to prevent ongoing compute charges..."
aws ec2 stop-instances --instance-ids $ID | Out-Null
Write-Host "Instance $ID successfully stopped!"
