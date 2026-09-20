$ErrorActionPreference = "Stop"

$AMI = "ami-0db16d2662b105f73"   # Deep Learning AMI
$TYPE = "g4dn.xlarge"
$KEY = "face-attendance"
$PEM = "$env:USERPROFILE\.ssh\face-attendance.pem"
$SG = "sg-0dca629c63e847a32"     
$NAME = "lunar-cnn-training"

# --- 1. Launch ---
Write-Host "Launching $TYPE ..."
$ID = aws ec2 run-instances `
    --image-id $AMI --instance-type $TYPE `
    --key-name $KEY --security-group-ids $SG `
    --block-device-mappings '[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":60,\"VolumeType\":\"gp3\"}}]' `
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME}]" `
    --query "Instances[0].InstanceId" --output text
Write-Host "Instance: $ID"

aws ec2 wait instance-running --instance-ids $ID
$IP = aws ec2 describe-instances --instance-ids $ID `
    --query "Reservations[0].Instances[0].PublicIpAddress" --output text
Write-Host "Public IP: $IP  (waiting 90s for SSH to come up)"
Start-Sleep -Seconds 90

$SSH = "ssh -i `"$PEM`" -o StrictHostKeyChecking=accept-new ubuntu@$IP"

# --- 2. Upload dataset + training script ---
Write-Host "Uploading dataset and training script ..."
scp -i "$PEM" -o StrictHostKeyChecking=accept-new "C:\Users\hp\Desktop\training image\Train\Train\train_images.zip" ubuntu@${IP}:~
scp -i "$PEM" -o StrictHostKeyChecking=accept-new "C:\Users\hp\Desktop\training image\Test\Test\eval_images.zip" ubuntu@${IP}:~
scp -i "$PEM" -o StrictHostKeyChecking=accept-new "C:\Users\hp\Desktop\training image\train_metadata.csv" ubuntu@${IP}:~
scp -i "$PEM" -o StrictHostKeyChecking=accept-new "C:\Users\hp\Desktop\training image\Test\Test\test_metadata.csv" ubuntu@${IP}:~
scp -i "$PEM" -o StrictHostKeyChecking=accept-new "train_lunar.py" ubuntu@${IP}:~

# --- 3. Train on the GPU ---
Write-Host "Setting up and training on EC2 (output streams below) ..."
$cmd = @"
unzip -q train_images.zip
unzip -q eval_images.zip
source activate pytorch 2>/dev/null || source /opt/pytorch/bin/activate 2>/dev/null || true
python3 -m pip install transformers peft timm scikit-learn pandas tqdm
python3 train_lunar.py
python3 train_lunar.py predict
"@
Invoke-Expression "$SSH '$cmd'"

# --- 4. Download results ---
Write-Host "Downloading lunar_model_output/ ..."
scp -i "$PEM" -r ubuntu@${IP}:~/lunar_model_output .

# --- 5. TERMINATE ---
Write-Host "Terminating $ID ..."
aws ec2 terminate-instances --instance-ids $ID | Out-Null
aws ec2 wait instance-terminated --instance-ids $ID
Write-Host "Done. Model + report are in .\lunar_model_output\"
