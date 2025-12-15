#!/usr/bin/env python3
"""
Create RunPod with SSH access
"""
import runpod
import time

runpod.api_key = 'rpa_29GS7ZOVYZRB5D2XFTZNBU9JQTOGNN3AEZC8KJR14ahqwp'

print("="*60)
print("STEP 1: Terminating old pods without SSH")
print("="*60)

# Terminate existing pods
pods = runpod.get_pods()
for pod in pods:
    print(f"Terminating pod: {pod['id']}")
    runpod.terminate_pod(pod['id'])

print("\n✓ Old pods terminated")
# time.sleep(3)

# print("\n" + "="*60)
# print("STEP 2: Finding RTX 6000 Ada")x``
# print("="*60)

# gpus = runpod.get_gpus()
# if not gpus:
#     print("No GPUs available!")
#     exit(1)

# # Find RTX 6000 Ada (also known as RTX Pro 6000)
# target_gpu = None
# for gpu in gpus:
#     if '6000' in gpu['displayName'] or 'RTX 6000' in gpu['displayName']:
#         target_gpu = gpu
#         break

# if not target_gpu:
#     print("RTX 6000 not available! Available GPUs:")
#     for idx, gpu in enumerate(gpus[:5], 1):
#         price = gpu.get('lowestPrice', {}).get('minimumBidPrice', 'N/A')
#         print(f"{idx}. {gpu['displayName']} - ${price}/hr")
#     print("\nUsing first available GPU instead...")
#     target_gpu = gpus[0]

# price = target_gpu.get('lowestPrice', {}).get('minimumBidPrice', 'N/A')
# print(f"✓ Selected: {target_gpu['displayName']} - ${price}/hr")

# print("\n" + "="*60)
# print("STEP 3: Creating new pod with SSH enabled")
# print("="*60)

# # Read SSH public key
# ssh_public_key = None
# ssh_key_path = '/home/codespace/.ssh/id_ed25519.pub'
# try:
#     with open(ssh_key_path, 'r') as f:
#         ssh_public_key = f.read().strip()
#     print(f"✓ Loaded SSH public key")
# except FileNotFoundError:
#     print(f"⚠ SSH key not found, generating one...")
#     import subprocess
#     subprocess.run(['ssh-keygen', '-t', 'ed25519', '-f', '/home/codespace/.ssh/id_ed25519', '-N', ''], check=True)
#     with open(ssh_key_path, 'r') as f:
#         ssh_public_key = f.read().strip()
#     print(f"✓ Generated and loaded SSH public key")

# # Create pod with SSH key
# pod = runpod.create_pod(
#     name="heartbeat-ssh",
#     image_name="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
#     gpu_type_id=target_gpu['id'],
#     cloud_type="SECURE",
#     volume_in_gb=50,
#     container_disk_in_gb=50,  # Increase container disk to 50GB to prevent disk full errors
#     ports="22/tcp,8888/http,6006/http",  # SSH + Jupyter + TensorBoard
#     env={
#         "PUBLIC_KEY": ssh_public_key
#     }
# )

# print(f"✓ Pod created: {pod['id']}")
# print("\nWaiting for pod to start (this takes ~60 seconds)...")

# # Wait for pod to be ready
# for i in range(30):
#     time.sleep(5)
#     try:
#         pod_info = runpod.get_pod(pod['id'])
#         runtime = pod_info.get('runtime', {})
        
#         # Look for SSH port (22)
#         ports = runtime.get('ports', [])
#         ssh_port = None
#         ssh_host = None
        
#         for port_info in ports:
#             if port_info.get('privatePort') == 22:
#                 ssh_host = port_info.get('ip')
#                 ssh_port = port_info.get('publicPort')
#                 break
        
#         if ssh_host and ssh_port:
#             print("\n" + "="*60)
#             print("✓ POD IS READY!")
#             print("="*60)
#             print(f"\nSSH Connection:")
#             print(f"  ssh root@{ssh_host} -p {ssh_port}")
#             print(f"\n🔑 Authentication: SSH Key (passwordless)")
#             print("\nNext steps:")
#             print("1. Copy the SSH command above")
#             print("2. Connect (no password needed)")
#             print("3. Run setup commands")
#             print("="*60)
#             break
#         else:
#             print(f"  Waiting... ({i*5}s)")
#     except:
#         print(f"  Starting... ({i*5}s)")
# else:
#     print("\n⚠ Timeout waiting for SSH. Check pod manually:")
#     print("  python get_pod_info.py")
