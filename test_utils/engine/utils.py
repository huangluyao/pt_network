import time
from getpass import getuser
from socket import gethostname


def get_host_info():
    return f'{getuser()}@{gethostname()}'

def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())
