from .client import ClientInfo


def expected_duration(c: ClientInfo) -> float:
    if c.estimated_duration and c.estimated_duration > 0:
        return float(c.estimated_duration)
    data = float(c.data_size or 0.0)
    comp = data / max(1e-06, float(c.compute_speed or 1.0))
    net = data / 1000.0 * (1.0 / max(1e-06, float(c.channel_quality or 1.0)))
    return comp + net
