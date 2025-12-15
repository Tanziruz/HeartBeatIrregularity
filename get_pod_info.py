#!/usr/bin/env python3
import runpod
import json

runpod.api_key = 'rpa_29GS7ZOVYZRB5D2XFTZNBU9JQTOGNN3AEZC8KJR14ahqwp'

pods = runpod.get_pods()

print("="*60)
print("YOUR RUNPOD INSTANCES")
print("="*60)

for pod in pods:
    print(f"\n🖥️  Pod: {pod['id']}")
    print(f"   Name: {pod.get('name', 'N/A')}")
    print(f"   Status: {pod.get('desiredStatus', 'unknown')}")
    
    runtime = pod.get('runtime', {})
    
    # Check if running
    uptime = runtime.get('uptimeInSeconds', 0)
    if uptime > 0:
        print(f"   ✓ Running (uptime: {uptime}s)")
    else:
        print(f"   ⚠ Not running yet")
    
    # SSH info
    ssh_host = runtime.get('sshHost')
    ssh_port = runtime.get('sshPort')
    
    if ssh_host and ssh_port:
        print(f"\n   📡 SSH Connection:")
        print(f"   ssh root@{ssh_host} -p {ssh_port}")
    else:
        print(f"\n   ⚠ SSH not ready yet - pod may still be starting")
    
    # Port info
    ports = runtime.get('ports', [])
    if ports:
        print(f"\n   🌐 Web Ports:")
        for port_info in ports:
            private = port_info.get('privatePort')
            public_url = port_info.get('publicUrl', 'Not available')
            print(f"   Port {private}: {public_url}")
    
    print()

print("="*60)
print("\nTo connect to a pod:")
print("1. Copy the SSH command above")
print("2. Paste it in your terminal")
print("3. Type 'yes' if asked about host key")
print("="*60)
