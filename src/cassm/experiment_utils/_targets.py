from tueplots import bundles

from ._tueplots_bundles import icml as icml_bundle
from ._tueplots_bundles import jmlr as jmlr_bundle
from ._tueplots_bundles import neurips as neurips_bundle

_tueplots_bundles = {
    "icml": icml_bundle,
    "neurips": neurips_bundle,
    "thesis": neurips_bundle,
    "jmlr": jmlr_bundle,
    "beamer": bundles.beamer_moml,
}
