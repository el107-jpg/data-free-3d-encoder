"""Device telemetry and short duty windows, including diagnostics and backwards."""
import ctypes as C
import json
import time
import torch
from torch.utils._python_dispatch import TorchDispatchMode

class Util(C.Structure):
    _fields_=[("gpu",C.c_uint),("memory",C.c_uint)]
class Memory(C.Structure):
    _fields_=[("total",C.c_ulonglong),("free",C.c_ulonglong),("used",C.c_ulonglong)]

class GpuBudget(TorchDispatchMode):
    def __init__(self,path):
        self.lib=C.CDLL("libnvidia-ml.so.1")
        self.check(self.lib.nvmlInit_v2())
        self.handle=C.c_void_p()
        self.check(self.lib.nvmlDeviceGetHandleByIndex_v2(0,C.byref(self.handle)))
        self.log=path.open("a",buffering=1)
        self.next_read=0.
        self.window=time.monotonic()
        self.read(clear=True)
        torch.cuda.set_per_process_memory_fraction(.70,0)

    @staticmethod
    def check(rc):
        if rc:raise RuntimeError(f'GPU telemetry unavailable (NVML {rc})')

    def read(self,clear=False):
        released=False
        while True:
            util=Util();mem=Memory()
            self.check(self.lib.nvmlDeviceGetUtilizationRates(self.handle,C.byref(util)))
            self.check(self.lib.nvmlDeviceGetMemoryInfo(self.handle,C.byref(mem)))
            busy=util.gpu;used=mem.used/mem.total
            self.log.write(json.dumps(dict(t=time.time(),util_pct=busy,used_mib=mem.used//2**20,total_mib=mem.total//2**20))+'\n')
            if used>.76 and not released and torch.cuda.is_initialized():
                torch.cuda.empty_cache();released=True
            if busy <= (65 if clear else 74) and used < .77:break
            clear=True
            time.sleep(.2)
        self.next_read=time.monotonic()+.20

    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        now=time.monotonic()
        if now-self.window>=.012:
            torch.cuda.synchronize()
            end=time.monotonic()
            # Synchronous launches plus short work windows avoid a queued backlog during idle
            # periods. Aim below the user's ceiling to leave tolerance for the display.
            time.sleep(min(end-self.window,.12)*.70)
            if time.monotonic()>=self.next_read:self.read()
            self.window=time.monotonic()
        return func(*args,**(kwargs or {}))
