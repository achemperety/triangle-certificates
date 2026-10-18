"""tricert: exact computations for "Triangles and certificates of non-3-colourability"."""
from . import ring, graphs, components, certificates, kneser, icosahedral  # noqa: F401
from .components import (triangles, triangle_components, propagate, compression_count, brute_force_count,  # noqa
                         every_edge_in_triangle, component_colourable)
from .certificates import (vieta_certificate, check_certificate, propagation_certificate,  # noqa: F401
                           floor_functional_check, ns_degree_f2, ns_certificate_exists_f2)

__version__ = "1.0.0"
