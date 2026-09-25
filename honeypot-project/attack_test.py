import paramiko
import time

target_host = "127.0.0.1"
target_port = 2222

# Fake users aur passwords ki list
test_credentials = [
    ("user1", "pass123"),
    ("admin_test", "adminpass"),
    ("hacker", "p@ssword"),
    ("guest", "guest123"),
    ("root_user", "root1234")
]

print("Starting SSH brute-force simulation...")

for user, pwd in test_credentials:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        print(f"Attempting login: User={user}, Password={pwd}")
        ssh.connect(target_host, port=target_port, username=user, password=pwd, timeout=2)
        ssh.close()
    except Exception as e:
        # Decoy SSH hamesha connection refuse/close karega login store karne ke baad
        pass
    
    time.sleep(1)  # 1 second gap

print("All test attempts finished!")