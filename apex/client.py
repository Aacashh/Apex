from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class ClientInfo:
    id: int
    data_size: int
    last_loss: float = 0.0
    grad_norm: float = 0.0
    compute_speed: float = 1.0
    dp_epsilon_remaining: float = 10.0
    dp_epsilon_used: float = 0.0
    participation_count: int = 0
    position: Optional[List[float]] = None
    channel_quality: float = 1.0
    estimated_duration: float = 0.0
    energy_rate: float = 1.0
    battery_capacity: float = 100.0
    region_carbon_g_per_kwh: float = 400.0
    estimated_energy: float = 0.0
    estimated_bytes: float = 0.0
    tier: Optional[int] = None
    utility_estimate: float = 0.0
    last_selected_round: int = -1
    label_histogram: Optional[Dict[int, int]] = None
    model_arch: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)
