# AB Advisor tests
import os

# Keep numeric libraries from oversubscribing CI / cloud CPUs.
os.environ.setdefault("OMP_NUM_THREADS", "1")
