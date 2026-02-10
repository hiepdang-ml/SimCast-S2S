import torch


class NamedModel:

    @property
    def name(self) -> str:
        if isinstance(self, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            name: str = self.module.__class__.__name__.lower()
        else:
            name: str = self.__class__.__name__.lower()

        name = name.replace("_","")
        name = "diffusion" if name == "unetdenoiser" else name
        return name
