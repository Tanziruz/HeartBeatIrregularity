#!/usr/bin/env python3
"""
Get SSH credentials for RunPod
"""
import runpod
import json

runpod.api_key = 'rpa_29GS7ZOVYZRB5D2XFTZNBU9JQTOGNN3AEZC8KJR14ahqwp'

pods = runpod.get_pods()

if not pods:
    print("No pods found!")
    exit(1)

print("="*60)
print("SSH CONNECTION INFO")
print("="*60)

for pod in pods:
    print(f"\n🖥️  Pod: {pod['id']}")
    print(f"   Name: {pod.get('name', 'N/A')}")
    
    runtime = pod.get('runtime', {})
    
    # Get SSH port
    ssh_host = None
    ssh_port = None
    
    for port_info in runtime.get('ports', []):
        if port_info.get('privatePort') == 22:
            ssh_host = port_info.get('ip')
            ssh_port = port_info.get('publicPort')
            break
    
    if ssh_host and ssh_port:
        print(f"\n   📡 SSH Connection:")
        print(f"   ssh root@{ssh_host} -p {ssh_port}")
        
        # Check for password in environment or other fields
        env = pod.get('env', [])
        print(f"\n   🔑 Authentication:")
        print(f"   - RunPod uses SSH keys by default")
        print(f"   - Password-based login is typically disabled")
        
        # Check if there's any auth info
        if 'password' in str(pod).lower():
            print(f"\n   Raw pod data (check for password):")
            print(json.dumps(pod, indent=2))
        else:
            print(f"\n   💡 To connect without password:")
            print(f"   1. Add your SSH key to RunPod account")
            print(f"   2. Or use RunPod web terminal (no password needed)")
            print(f"\n   Or try connecting - it may be passwordless:")
            print(f"   ssh root@{ssh_host} -p {ssh_port}")
    else:
        print(f"   ⚠ SSH not ready yet")

print("\n" + "="*60)
