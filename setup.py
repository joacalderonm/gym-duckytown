import subprocess
import sys
import os

def install_requirements():
    if not os.path.exists('requirements.txt'):
        print("requirements.txt file is doesn't exists")
        sys.exit(1)
    
    print("Instalando las dependencias desde requirements.txt...")
    result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Error installing dependencies")
        print(result.stderr)
        sys.exit(1)
    else:
        print("Correct installation")

def run_additional_script():
    print("Modifying librarie.")
    result = subprocess.run([sys.executable, 'contracts_modify.py'], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Error!")
        print(result.stderr)
        sys.exit(1)
    else:
        print("Correct modification")

def main():
    install_requirements()
    run_additional_script()
    print("You are ready to work!")

if __name__ == '__main__':
    main()