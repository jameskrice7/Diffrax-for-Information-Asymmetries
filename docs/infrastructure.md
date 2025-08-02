# Infrastructure

Finax provides helpers to configure experiments and leverage hardware accelerators.

## Device Utilities
- `finax.infrastructure.devices.available_devices` lists JAX devices visible to the runtime.
- `finax.infrastructure.devices.default_device` selects a GPU or TPU when available.
- `finax.infrastructure.devices.to_device` moves arrays to the chosen device.

```python
from finax.infrastructure.devices import to_device
import jax.numpy as jnp
x = to_device(jnp.ones((2, 2)))
```

## Configuration
- `finax.infrastructure.config.load_config` loads JSON configuration files for reproducible pipelines.
