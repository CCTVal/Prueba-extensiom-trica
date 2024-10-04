import torch
import subprocess
import sys
import pkg_resources

def get_gpu_info():
    if torch.cuda.is_available():
        return f"GPU: {torch.cuda.get_device_name(0)}"
    else:
        return "GPU no disponible"

def get_cuda_version():
    if torch.cuda.is_available():
        return f"CUDA Version: {torch.version.cuda}"
    else:
        return "CUDA no disponible"

def get_pytorch_version():
    return f"PyTorch Version: {torch.__version__}"

def get_ultralytics_version():
    try:
        return f"Ultralytics Version: {pkg_resources.get_distribution('ultralytics').version}"
    except pkg_resources.DistributionNotFound:
        return "Ultralytics no encontrado o no instalado correctamente"

def get_python_version():
    return f"Python Version: {sys.version.split()[0]}"

def get_os_info():
    try:
        if sys.platform.startswith('win'):
            os_info = subprocess.check_output('systeminfo | findstr /B /C:"OS Name" /C:"OS Version"', shell=True).decode()
        elif sys.platform.startswith('linux'):
            os_info = subprocess.check_output(['lsb_release', '-a']).decode()
        elif sys.platform.startswith('darwin'):
            os_info = subprocess.check_output(['sw_vers']).decode()
        else:
            os_info = "Sistema operativo no reconocido"
        return f"OS Info:\n{os_info}"
    except:
        return "No se pudo obtener información del sistema operativo"

if __name__ == "__main__":
    print(get_gpu_info())
    print(get_cuda_version())
    print(get_pytorch_version())
    print(get_ultralytics_version())
    print(get_python_version())
    print(get_os_info())