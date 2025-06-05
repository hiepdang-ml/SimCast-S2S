from functools import cached_property

class HasModelName:
    @cached_property
    def model_name(self) -> str:
        base_name = self.net.__class__.__name__.lower()
        if hasattr(self, "tp"):
            return f"{base_name}_{self.tp.lower()}"
        return base_name

