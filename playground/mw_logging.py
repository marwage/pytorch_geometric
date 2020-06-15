import logging
import time
import torch


def log_gpu_memory(when, start=0):
    mib = pow(2, 20)
    if start == 0:
        since_start = 0
    else:
        since_start = time.time() - start
    stats = torch.cuda.memory_stats()
    logging.debug("{:.2f}s:{}:active.current {:.2f}MiB;allocated.current {:.2f}MiB;reserved.current {:.2f}MiB".format(since_start, when, stats["active_bytes.all.current"] / mib, stats["allocated_bytes.all.current"] / mib, stats["reserved_bytes.all.current"] / mib))
    logging.debug("{:.2f}s:{}:active.peak {:.2f}MiB;allocated.peak {:.2f}MiB;reserved.peak {:.2f}MiB".format(since_start, when, stats["active_bytes.all.peak"] / mib, stats["allocated_bytes.all.peak"] / mib, stats["reserved_bytes.all.peak"] / mib))

def log_tensor(tensor, name="?"):
    logging.debug("{}:shape {};data type {};pointer {};size {}".format(name, tensor.size(), tensor.dtype, tensor.storage().data_ptr(), tensor.storage().size()))
    if tensor.grad is not None:
        logging.debug("{}:shape {};data type {};pointer {};size {}".format(name + ".grad", tensor.grad.size(), tensor.grad.dtype, tensor.grad.storage().data_ptr(), tensor.grad.storage().size()))
