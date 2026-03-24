from typing import Tuple

class Well:
    """Represents a source or sink in the reservoir."""
    
    def __init__(self, name: str, location: Tuple[int, int, int], rate: float = 0.0, bhp: float = None, 
                 well_radius: float = 0.25, skin: float = 0.0):
        """
        :param name: Well identifier
        :param location: (i, j, k) zero-indexed grid coordinates
        :param rate: Specified flow rate (STB/d), negative=production, positive=injection
        :param bhp: Specified bottom hole pressure (psi)
        :param well_radius: Wellbore radius (ft)
        :param skin: Skin factor (dimensionless)
        """
        self.name = name
        self.location = location
        self.rate = rate
        self.bhp = bhp
        self.well_radius = well_radius
        self.skin = skin
        
        if rate == 0.0 and bhp is None:
            raise ValueError("Well must have either a rate or a BHP specification")

    @property
    def is_producer(self) -> bool:
        return self.rate < 0

    @property
    def is_injector(self) -> bool:
        return self.rate > 0
