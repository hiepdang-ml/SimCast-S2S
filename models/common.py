import torch


class NamedModel:

    @property
    def name(self) -> str:
        if isinstance(self, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            basename: str = self.module.__class__.__name__.lower()
        else:
            basename: str = self.__class__.__name__.lower()
        
        if hasattr(self, "tp"):
            return f"{basename}{self.tp.lower()}"
        else:
            return basename
        
