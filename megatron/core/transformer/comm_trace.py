# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Communication trace logging for all-to-all and related operations."""

import json
import logging
import os
import time
from typing import Optional, Dict, Any

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class CommTraceLogger:
    """Logger for communication traces in JSON format."""

    def __init__(self, log_dir: Optional[str] = None, enabled: bool = True):
        """Initialize the communication trace logger.
        
        Args:
            log_dir (str, optional): Directory to save trace files. If None, uses environment variable.
            enabled (bool): Whether tracing is enabled. Defaults to True.
        """
        self.enabled = enabled
        self.log_dir = log_dir or os.environ.get('COMM_TRACE_LOG_DIR', None)
        self.traces: list = []
        
        if self.enabled and self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            try:
                rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            except:
                rank = 0
            self.log_file = os.path.join(self.log_dir, f'comm_trace_rank{rank}.jsonl')
            
            # Clear previous logs
            with open(self.log_file, 'w') as f:
                f.write('')
    
    def log_alltoall_event(self,
                          op_type: str,
                          phase: str,
                          layer_id: int,
                          input_shape: tuple,
                          output_shape: tuple,
                          bytes_sent: int,
                          elapsed_ms: float,
                          group_size: int = 1,
                          metadata: Optional[Dict[str, Any]] = None):
        """Log an all-to-all event.
        
        Args:
            op_type (str): Operation type (e.g., 'alltoall', 'alltoall_sp2hp', 'alltoall_hp2sp')
            phase (str): Phase of operation ('FW' or 'BW')
            layer_id (int): Layer number or index
            input_shape (tuple): Shape of input tensor
            output_shape (tuple): Shape of output tensor
            bytes_sent (int): Number of bytes sent
            elapsed_ms (float): Elapsed time in milliseconds
            group_size (int): Size of process group
            metadata (dict, optional): Additional metadata
        """
        if not self.enabled or not self.log_dir:
            return
        
        try:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        except:
            rank = 0
        
        event = {
            'timestamp': time.time(),
            'rank': rank,
            'op_type': op_type,
            'phase': phase,
            'layer_id': layer_id,
            'input_shape': list(input_shape),
            'output_shape': list(output_shape),
            'bytes_sent': bytes_sent,
            'elapsed_ms': elapsed_ms,
            'group_size': group_size,
            'metadata': metadata or {},
        }
        
        self.traces.append(event)
        self._write_trace(event)
    
    def _write_trace(self, event: Dict[str, Any]):
        """Write a trace event to the log file."""
        if not self.enabled or not self.log_dir:
            return
        
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(event) + '\n')
        except Exception as e:
            logger.warning(f"Failed to write trace: {e}")
    
    def log_communication_info(self, info: str):
        """Log general communication information.
        
        Args:
            info (str): Information string to log
        """
        if not self.enabled:
            return
        
        logger.info(f"[COMM_TRACE] {info}")


# Global instance
_comm_trace_logger: Optional[CommTraceLogger] = None


def get_comm_trace_logger() -> CommTraceLogger:
    """Get the global communication trace logger instance."""
    global _comm_trace_logger
    if _comm_trace_logger is None:
        _comm_trace_logger = CommTraceLogger()
    return _comm_trace_logger


def set_comm_trace_enabled(enabled: bool):
    """Enable or disable communication tracing."""
    global _comm_trace_logger
    if _comm_trace_logger is None:
        _comm_trace_logger = CommTraceLogger(enabled=enabled)
    else:
        _comm_trace_logger.enabled = enabled
