import pathlib
import matplotlib.pyplot as plt
import torch


class _BasePlotter:

    def __init__(self) -> None:
        self.destination_directory: pathlib.Path = pathlib.Path('./results')
        self.destination_directory.mkdir(parents=True, exist_ok=True)

    def plot_layer(
        self, 
        ax, data: torch.Tensor, 
        coords: tuple[torch.Tensor, torch.Tensor], 
        title: str,
        cmap: str,
        vmin: float, vmax: float, 
    ) -> None:
        im = ax.pcolormesh(
            coords[1], coords[0], data,
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            shading='nearest',
        )
        ax.set_title(title, fontsize=12)
        cbar = ax.figure.colorbar(im, ax=ax, orientation='vertical', fraction=0.035, pad=0.04)
        cbar.ax.tick_params(labelsize=10)
        self._clean_axes(ax)

    def add_landmask(self, axs, landmask: torch.Tensor, coords: tuple[torch.Tensor, torch.Tensor]) -> None:
        for ax in axs:
            ax.contour(
                coords[1], coords[0], landmask,
                levels=[0.5], colors='black', linewidths=1,
            )
            self._clean_axes(ax)

    @staticmethod
    def _clean_axes(ax):
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(''); ax.set_ylabel('')


class PredictionPlotter(_BasePlotter):

    def plot(
        self,
        groundtruth_frame: torch.Tensor,
        prediction_frame: torch.Tensor,
        error_frame: torch.Tensor,
        landmask: torch.Tensor,
        coordinates: tuple[torch.Tensor, torch.Tensor],
        title: str,
        filename: str,
    ) -> None:

        assert groundtruth_frame.shape == prediction_frame.shape == error_frame.shape == landmask.shape == (192, 288)

        groundtruth_frame = groundtruth_frame.cpu()
        prediction_frame = prediction_frame.cpu()
        error_frame = error_frame.cpu()
        landmask = landmask.cpu()

        H: int = groundtruth_frame.shape[0]
        W: int = groundtruth_frame.shape[1]
        aspect_ratio: float = H / W
        figwidth: float = 5.5

        fig, axs = plt.subplots(3, 1, figsize=(figwidth, 3 * figwidth * aspect_ratio))
        q: float = 0.95
        limit: float = max(groundtruth_frame.quantile(q=q).item(), groundtruth_frame.neg().quantile(q=q).item())
        vmax: float = limit
        vmin: float = limit * (-1)

        self.plot_layer(ax=axs[0], data=groundtruth_frame, coords=coordinates, title="Groundtruth", cmap="RdBu", vmin=vmin, vmax=vmax)
        self.plot_layer(ax=axs[1], data=prediction_frame, coords=coordinates, title="Prediction", cmap="RdBu", vmin=vmin, vmax=vmax)
        self.plot_layer(ax=axs[2], data=error_frame, coords=coordinates, title="Error Map", cmap="RdBu", vmin=vmin, vmax=vmax)
        self.add_landmask(axs=axs, landmask=landmask, coords=coordinates)

        fig.subplots_adjust(left=0.01, right=0.97, bottom=0.05, top=0.87, hspace=0.15)
        fig.suptitle(title, fontsize=12)

        fig.savefig(self.destination_directory.joinpath(filename), bbox_inches="tight")
        plt.close(fig)


class MetricPlotter(_BasePlotter):

    def plot(
        self,
        mae_frame: torch.Tensor,
        rsquared_frame: torch.Tensor,
        landmask: torch.Tensor,
        coordinates: tuple[torch.Tensor, torch.Tensor],
        title: str,
        filename: str,
    ) -> None:
        
        assert mae_frame.shape == rsquared_frame.shape == landmask.shape == (192, 288)

        mae_frame = mae_frame.cpu()
        rsquared_frame = rsquared_frame.cpu()
        landmask = landmask.cpu()

        H: int = rsquared_frame.shape[0]
        W: int = rsquared_frame.shape[1]
        aspect_ratio: float = H / W
        figwidth: float = 5.5

        fig, axs = plt.subplots(2, 1, figsize=(figwidth, 2 * figwidth * aspect_ratio))
        self.plot_layer(ax=axs[0], data=mae_frame, coords=coordinates, title="MAE Map", cmap="Oranges", vmin=0., vmax=0.05)
        self.plot_layer(ax=axs[1], data=rsquared_frame, coords=coordinates, title="R-squared Map", cmap="Blues", vmin=0., vmax=1.)
        self.add_landmask(axs=axs, landmask=landmask, coords=coordinates)

        fig.subplots_adjust(left=0.01, right=0.97, bottom=0.05, top=0.88, hspace=0.1)
        fig.suptitle(title, fontsize=12)

        fig.savefig(self.destination_directory.joinpath(filename), bbox_inches="tight")
        plt.close(fig)



