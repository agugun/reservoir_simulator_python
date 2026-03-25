from typing import Tuple

class Well:
    """Represents a source or sink in the reservoir."""
    
    def __init__(self, name: str, location: Tuple[int, int, int], rate: float = 0.0, bhp: float = None,
                 well_radius: float = 0.25, skin: float = 0.0,
                 orat: float = None, grat: float = None, gir: float = None):
        """
        :param name:        Well identifier
        :param location:    (i, j, k) zero-indexed grid coordinates
        :param rate:        Specified flow rate (STB/d), negative=production, positive=injection
        :param bhp:         Bottom hole pressure constraint (psia)
        :param well_radius: Wellbore radius (ft)
        :param skin:        Skin factor (dimensionless)
        :param orat:        Target oil production rate (STB/day) — surface conditions
        :param grat:        Target gas production rate (MSCF/day) — surface conditions
        :param gir:         Target gas injection rate (MSCF/day) — surface conditions
        """
        self.name = name
        self.location = location
        self.rate = rate
        self.bhp = bhp
        self.well_radius = well_radius
        self.skin = skin
        self.orat = orat   # surface oil production target (STB/day)
        self.grat = grat   # surface gas production target (MSCF/day)
        self.gir  = gir    # gas injection target (MSCF/day)
        
        if rate == 0.0 and bhp is None and orat is None and gir is None:
            raise ValueError("Well must have either a rate, BHP, orat, or gir specification")

    @property
    def is_producer(self) -> bool:
        return self.rate < 0 or (self.orat is not None and self.orat > 0)

    @property
    def is_injector(self) -> bool:
        return self.rate > 0 or (self.gir is not None and self.gir > 0)

