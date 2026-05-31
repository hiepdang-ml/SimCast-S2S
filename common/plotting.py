import pathlib
from typing import Any

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import torch


class _BasePlotter:

    def __init__(self, dirpath: str) -> None:
        self.dirpath: pathlib.Path = pathlib.Path(dirpath)
        self.dirpath.mkdir(parents=True, exist_ok=True)

    def plot_layer(
        self,
        ax, data: torch.Tensor,
        coords: tuple[torch.Tensor, torch.Tensor],
        tropical_lats: tuple[float, float],
        title: str, cmap: str, vmin: float, vmax: float,
        data_crs: Any | None = None,
    ) -> None:
        levels = np.linspace(vmin, vmax, 17)
        im = ax.contourf(
            coords[1], coords[0], data,
            levels=levels, cmap=cmap, vmin=vmin, vmax=vmax,
            extend="both",
            transform=data_crs,
        )
        ax.set_title(title, fontsize=12)
        if data_crs is None:
            ax.axhline(y=tropical_lats[0], color="black", linewidth=0.8, linestyle="--")
            ax.axhline(y=tropical_lats[-1], color="black", linewidth=0.8, linestyle="--")
        else:
            longitudes = coords[1]
            for latitude in tropical_lats:
                ax.plot(
                    longitudes,
                    torch.full_like(longitudes, fill_value=latitude),
                    color="black",
                    linewidth=0.8,
                    linestyle="--",
                    transform=data_crs,
                )
            ax.set_global()
        cbar = ax.figure.colorbar(im, ax=ax, orientation='vertical', fraction=0.025, pad=0.04)
        cbar.ax.tick_params(labelsize=10)
        self._clean_axes(ax)

    def add_landmask(
        self,
        axs,
        landmask: torch.Tensor,
        coords: tuple[torch.Tensor, torch.Tensor],
        data_crs: Any | None = None,
    ) -> None:
        for ax in axs:
            ax.contour(
                coords[1],
                coords[0],
                landmask,
                levels=[0.5],
                colors='black',
                linewidths=1,
                transform=data_crs,
            )
            if data_crs is not None:
                ax.set_global()
            self._clean_axes(ax)

    @staticmethod
    def _clean_axes(ax):
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(''); ax.set_ylabel('') # noqa: E702


class PredictionPlotter(_BasePlotter):

    def plot(
        self,
        groundtruth_frame: torch.Tensor | None,
        prediction_frame: torch.Tensor | None,
        error_frame: torch.Tensor | None,
        uncertainty_frame: torch.Tensor | None,
        landmask: torch.Tensor,
        tropical_lats: tuple[float, float],
        coordinates: tuple[torch.Tensor, torch.Tensor],
        title: str, filename: str,
        vlim: float | None,
        use_cartopy_projection: bool = False,
    ) -> None:

        for frame in (groundtruth_frame, prediction_frame, error_frame, uncertainty_frame):
            if frame is not None:
                assert frame.shape == landmask.shape

        plot_items: list[tuple[str, torch.Tensor]] = []
        if prediction_frame is not None:
            plot_items.append(("Prediction", prediction_frame.cpu()))
        if groundtruth_frame is not None:
            plot_items.append(("Groundtruth", groundtruth_frame.cpu()))
        if error_frame is not None:
            plot_items.append(("Error Map", error_frame.cpu()))
        if uncertainty_frame is not None:
            plot_items.append(("Uncertainty", uncertainty_frame.cpu()))
        if not plot_items:
            raise ValueError("At least one of groundtruth_frame, prediction_frame, error_frame, uncertainty_frame must be provided.")

        landmask = landmask.cpu()
        H: int = landmask.shape[0]
        W: int = landmask.shape[1]
        aspect_ratio: float = H / W
        subplot_width: float = 5.6
        subplot_height: float = subplot_width * aspect_ratio
        nrows: int = len(plot_items)
        projection: ccrs.Robinson | None = None
        data_crs: ccrs.PlateCarree | None = None
        if use_cartopy_projection:
            longitudes: torch.Tensor = coordinates[1]
            central_longitude: int = int((longitudes.min() + longitudes.max()).item() / 2.0)
            projection = ccrs.Robinson(central_longitude=central_longitude)
            data_crs = ccrs.PlateCarree()
        fig, axs = plt.subplots(
            nrows,
            1,
            figsize=(subplot_width, int((nrows + 0.1) * subplot_height)),
            subplot_kw={"projection": projection} if projection is not None else None,
        )
        if nrows == 1:
            axs = [axs]

        if vlim is None:
            assert len(plot_items) > 1  # should have groundtruth frame to reference
            q: float = 0.95
            reference_frame: torch.Tensor = plot_items[1][1]
            vlim: float = reference_frame.abs().quantile(q=q).item()
        else:
            assert vlim > 0

        for ax, (subplot_title, frame) in zip(axs, plot_items):
            if subplot_title == "Uncertainty":
                (cmap, vmin, vmax) = ("Blues", 0, vlim ** 2)
            else:
                (cmap, vmin, vmax) = ("RdBu", -vlim, vlim)
            self.plot_layer(
                ax=ax, data=frame, coords=coordinates, tropical_lats=tropical_lats,
                title="" if nrows == 1 else subplot_title,
                cmap=cmap, vmin=vmin, vmax=vmax,
                data_crs=data_crs,
            )
        self.add_landmask(axs=axs, landmask=landmask, coords=coordinates, data_crs=data_crs)

        top: float
        if nrows == 4:
            top = 0.88
        elif nrows == 3:
            top = 0.85
        else:
            top = 0.65

        fig.subplots_adjust(left=0.01, right=0.97, bottom=0.05, top=top, hspace=0.10)
        fig.suptitle(title, fontsize=12)
        fig.savefig(self.dirpath.joinpath(filename), bbox_inches="tight", dpi=500)
        plt.close(fig)


class DenoisingPlotter:

    def __init__(self) -> None:
        self.dirpath: pathlib.Path = pathlib.Path('./steps')
        self.dirpath.mkdir(parents=True, exist_ok=True)

    def plot(
        self,
        x0_x0_mae: list[float],
        xk_x0_mae: list[float],
        noise: list[float],
        filename: str,
    ) -> None:

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        assert len(x0_x0_mae) == len(xk_x0_mae) == len(noise)
        n_steps: int = len(x0_x0_mae)
        steps = list(range(1, n_steps + 1))

        axs[0].plot(
            steps, list(reversed(x0_x0_mae)),
            label=r"$| \hat{x}_0 - x_0 |$", color="tab:blue",
        )
        axs[0].plot(
            steps, list(reversed(xk_x0_mae)),
            label=r"$| \hat{x}_k - x_0 |$", color="tab:orange",
        )
        axs[0].set_xlabel("step")
        axs[0].set_ylabel("MAE")
        axs[0].legend()
        axs[0].grid(True, linestyle="--", alpha=0.5)
        axs[0].set_xlim(1, n_steps)
        axs[0].set_ylim(0., 2.)
        axs[0].invert_xaxis()

        axs[1].plot(
            steps, noise,
            label=r"$\sqrt{1 - \bar{\alpha}_k}$",
            color="tab:green",
        )
        axs[1].set_xlabel("step")
        axs[1].set_ylabel("noise")
        axs[1].legend()
        axs[1].grid(True, linestyle="--", alpha=0.5)
        axs[1].set_xlim(1, n_steps)
        axs[1].set_ylim(0., 1.)
        axs[1].invert_xaxis()

        fig.subplots_adjust(wspace=0.25, bottom=0.15, top=0.85)
        fig.savefig(self.dirpath.joinpath(filename), bbox_inches="tight", dpi=500)
        plt.close(fig)
