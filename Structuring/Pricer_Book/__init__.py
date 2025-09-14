# Structuring/Pricer_Book/__init__.py
from .option_bond import render as option_bond_ui
from .zero_coupon import render as zero_coupon_ui
from .fixed_floating_bond import render as fixed_floating_bond_ui
from .slb_carrefour_case import render as slb_carrefour_case_ui
from .convertible_bond import convertible_bond_ui  # la fonction s'appelle bien convertible_bond_ui

__all__ = [
    "option_bond_ui",
    "zero_coupon_ui",
    "fixed_floating_bond_ui",
    "slb_carrefour_case_ui",
    "convertible_bond_ui",
]
