import torch


class NamedModel:

    @property
    def name(self) -> str:
        name: str
        if isinstance(self, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            name = self.module.__class__.__name__.lower()
        else:
            name = self.__class__.__name__.lower()

        name = name.replace("_","")
        name = "diffusion" if name == "denoiser" else name
        return name
