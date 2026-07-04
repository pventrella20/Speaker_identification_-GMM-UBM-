"""GMM-UBM speaker identification system.

Modernized architecture: a thin domain layer (speaker/audio value objects),
infrastructure adapters (filesystem, audio splitting, model persistence),
a ml layer (feature extraction, MAP adaptation, classification strategy)
and an application layer of use-cases orchestrated by a CLI.
"""

__version__ = "2.0.0"
