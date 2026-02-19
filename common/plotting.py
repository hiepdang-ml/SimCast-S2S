import pathlib
import matplotlib.pyplot as plt
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
    ) -> None:
        im = ax.pcolormesh(
            coords[1], coords[0], data,
            cmap=cmap, vmin=vmin, vmax=vmax,
            shading='nearest',
        )
        ax.set_title(title, fontsize=12)
        ax.axhline(y=tropical_lats[0], color="black", linewidth=0.8, linestyle="--")
        ax.axhline(y=tropical_lats[-1], color="black", linewidth=0.8, linestyle="--")
        cbar = ax.figure.colorbar(im, ax=ax, orientation='vertical', fraction=0.035, pad=0.04)
        cbar.ax.tick_params(labelsize=10)
        self._clean_axes(ax)

    def add_landmask(self, axs, landmask: torch.Tensor, coords: tuple[torch.Tensor, torch.Tensor]) -> None:
        for ax in axs:
            ax.contour(coords[1], coords[0], landmask, levels=[0.5], colors='black', linewidths=1)
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
        fig, axs = plt.subplots(nrows, 1, figsize=(subplot_width, int((nrows + 0.6) * subplot_height)))
        if nrows == 1:
            axs = [axs]

        if vlim is None:
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
            )
        self.add_landmask(axs=axs, landmask=landmask, coords=coordinates)

        if nrows == 4:
            top: float = 0.89
        elif nrows == 3:
            top: float = 0.87
        elif nrows == 2:
            top: float = 0.82
        else:
            top: float = 0.75

        fig.subplots_adjust(left=0.01, right=0.97, bottom=0.05, top=top, hspace=0.14)
        fig.suptitle(title, fontsize=12)
        fig.savefig(self.dirpath.joinpath(filename), bbox_inches="tight", dpi=500)
        plt.close(fig)


class MetricPlotter(_BasePlotter):

    def plot(
        self,
        mae_frame: torch.Tensor,
        global_mae: torch.Tensor,
        tropical_mae: torch.Tensor,
        extratropical_mae: torch.Tensor,
        rsquared_frame: torch.Tensor,
        global_rsquared: torch.Tensor,
        tropical_rsquared: torch.Tensor,
        extratropical_rsquared: torch.Tensor,
        landmask: torch.Tensor,
        tropical_lats: tuple[float, float],
        coordinates: tuple[torch.Tensor, torch.Tensor],
        title: str,
        filename: str,
    ) -> None:

        assert mae_frame.shape == rsquared_frame.shape == landmask.shape

        mae_frame = mae_frame.cpu()
        rsquared_frame = rsquared_frame.cpu()
        landmask = landmask.cpu()

        H: int = rsquared_frame.shape[0]
        W: int = rsquared_frame.shape[1]
        aspect_ratio: float = H / W
        figwidth: float = 5.8

        fig, axs = plt.subplots(2, 1, figsize=(figwidth, 2 * figwidth * aspect_ratio))
        sub_title: str = (
            f"MAE Map\n"
            f"Global: {global_mae.item():.3f} - "
            f"Tropic: {tropical_mae.item():.3f} - "
            f"Extratropic: {extratropical_mae.item():.3f}"
        )
        self.plot_layer(
            ax=axs[0], data=mae_frame, coords=coordinates,
            tropical_lats=tropical_lats, title=sub_title, cmap="Oranges", vmin=0., vmax=0.05,
        )
        sub_title: str = (
            f"R-squared Map\n"
            f"Global: {global_rsquared.item():.3f} - "
            f"Tropic: {tropical_rsquared.item():.3f} - "
            f"Extratropic: {extratropical_rsquared.item():.3f}"
        )
        self.plot_layer(
            ax=axs[1], data=rsquared_frame, coords=coordinates,
            tropical_lats=tropical_lats, title=sub_title, cmap="Blues", vmin=0., vmax=0.6,
        )
        self.add_landmask(axs=axs, landmask=landmask, coords=coordinates)

        fig.subplots_adjust(left=0.01, right=0.97, bottom=0.05, top=0.88, hspace=0.18)
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
