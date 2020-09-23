import logging
import time
import torch
from torch_sparse import SparseStorage, SparseTensor

max_peak_allocated = 0
mib = pow(2, 20)
start = 0
last_current_active = 0
last_current_active_byte = 0


def log_gpu_memory(when):
    global start
    if start == 0:
        since_start = 0
    else:
        since_start = time.time() - start
    stats = torch.cuda.memory_stats()
    logging.debug("{:.2f}s:{}:active.current {:.2f}MiB;allocated.current {:.2f}MiB;reserved.current {:.2f}MiB".format(since_start, when, stats["active_bytes.all.current"] / mib, stats["allocated_bytes.all.current"] / mib, stats["reserved_bytes.all.current"] / mib))
    logging.debug("{:.2f}s:{}:active.peak {:.2f}MiB;allocated.peak {:.2f}MiB;reserved.peak {:.2f}MiB".format(since_start, when, stats["active_bytes.all.peak"] / mib, stats["allocated_bytes.all.peak"] / mib, stats["reserved_bytes.all.peak"] / mib))


def log_tensor(tensor, name="?"):
    if tensor is not None:
        if isinstance(tensor, SparseTensor):
            logging.debug("{}:shape {};data type {}; device {}".format(name, tensor.sizes(), tensor.dtype(), tensor.device()))
            log_sparse_storage(tensor.storage, name=name)
        else:
            logging.debug("{}:shape {};data type {};pointer {};size {}; device {}".format(name, tensor.size(), tensor.dtype, tensor.storage().data_ptr(), tensor.storage().size(), tensor.device))
            if tensor.grad is not None:
                logging.debug("{}:shape {};data type {};pointer {};size {}; device {}".format(name + ".grad", tensor.grad.size(), tensor.grad.dtype, tensor.grad.storage().data_ptr(), tensor.grad.storage().size(), tensor.device))


def log_peak_increase(where:str, device=None):
    global max_peak_allocated
    max_allocated = torch.cuda.max_memory_allocated(device=device)
    if max_allocated > max_peak_allocated:
        increase = (max_allocated - max_peak_allocated) / mib
        logging.debug("{}:increase of allocated_bytes.all.peak {:.2f}MiB".format(where, increase))
        max_peak_allocated = max_allocated
    else:
        logging.debug("{}:No increase of allocated_bytes.all.peak".format(where))


def log_sparse_storage(storage: SparseStorage, name="?"):
    log_tensor(storage._row, "{} row".format(name))
    log_tensor(storage._rowptr, "{} rowptr".format(name))
    log_tensor(storage._col, "{} column".format(name))
    log_tensor(storage._value, "{} value".format(name))
    log_tensor(storage._rowcount, "{} rowcount".format(name))
    log_tensor(storage._colptr, "{} colptr".format(name))
    log_tensor(storage._colcount, "{} colcount".format(name))
    log_tensor(storage._csr2csc, "{} csr2csc".format(name))
    log_tensor(storage._csc2csr, "{} csc2csr".format(name))


def backward_hook(module, grad_input, grad_output):
    log_peak_increase("in backward hook of {}".format(str(module)))
    for i, grad_in in enumerate(grad_input):
        log_tensor(grad_in, "grad_input {}".format(i))
    for i, grad_out in enumerate(grad_output):
        log_tensor(grad_out, "grad_output {}".format(i))


def set_start():
    global start
    start = time.time()


def log_timestamp(when: str):
    global start
    now = time.time() - start
    logging.debug("Timestamp {}: {:.6f}".format(when, now))

def log_current_active(where:str="?"):
    global last_current_active
    global last_current_active_byte
    stats = torch.cuda.memory_stats()
    current_active_byte =  stats["active_bytes.all.current"]
    current_active = current_active_byte / mib
    diff = current_active - last_current_active
    diff_byte = current_active_byte- last_current_active_byte
    logging.debug("{}:GPU.active.current {:.2f}MiB, diff {:.2f}MiB".format(where, current_active, diff))
    logging.debug("{}:GPU.active.current {}B, diff {}B".format(where, current_active_byte, diff_byte))
    last_current_active = current_active
    last_current_active_byte = current_active_byte
