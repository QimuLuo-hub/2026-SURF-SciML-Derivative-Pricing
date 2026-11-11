"""
Draw the poster's R3 figure: the American-put price slice at tau/T = 0.5.

It plots the DIRK reference, the Vanilla PINN (Arm C), the structure-aware
PINN (Arm D), and the d = 50 FC-HVI and DeepSets-HVI models, with a zoomed
inset near the strike. Without --no-show, an interactive Matplotlib window
opens; the current view can be saved from it.

The numerical data are read only from existing stored experiment artifacts.
No retraining, rescoring, or tuning is performed. By default every input is
read from this repository.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
TEAM_ROOT = PACKAGE.parent
DEFAULT_REFERENCE = (
    TEAM_ROOT / "numerical_and_pinn" / "02_dirk_reference" / "results" / "reference"
)
DEFAULT_ARM_C = TEAM_ROOT / "numerical_and_pinn" / "03_vanilla_pinn" / "results" / "poster_case"
DEFAULT_ARM_D = PACKAGE / "03_structure_aware_pinn" / "results" / "poster_case"
DEFAULT_HD_ROOT = (
    PACKAGE / "04_high_dimensional_hvi" / "results" / "hd3_matched_american_put_boundary_complete"
)
DEFAULT_OUTPUT = PACKAGE / "runs" / "r3_figure"


# ======================================================================
# Configuration
# ======================================================================

REGIME = "put_T100_s060_r005_q010"

COMMON_S_MIN = 0.5
COMMON_S_MAX = 1.5

TAU_TARGET = 0.5
HD_DIMENSION = 50
DISPLAY_MODES = ("normalized", "k1")

COLORS = {
    "reference": "#202020",
    "arm_c": "#A23B72",
    "arm_d": "#2A9D8F",
    "fc": "#2878B5",
    "deepsets": "#E68A2E",
}


# ======================================================================
# Data loading
# ======================================================================

def high_dimensional_surface_dirs(hd_root: Path) -> tuple[Path, Path]:
    """Return the frozen d50 FC-HVI and DeepSets-HVI visualization paths."""

    dimension_root = hd_root / "visualization" / f"d{HD_DIMENSION}"
    return dimension_root / "FC_HVI", dimension_root / "DEEPSETS_HVI"


def validate_k1_sources(source_dirs: list[Path]) -> None:
    """Require every plotted source to declare the matched strike K=1."""

    for source_dir in source_dirs:
        metadata_path = source_dir / "metadata.json"
        with metadata_path.open() as handle:
            metadata = json.load(handle)
        declared_strikes = []
        if "K" in metadata:
            declared_strikes.append(float(metadata["K"]))
        for section_name in ("asset_contract", "target_reduced_contract"):
            section = metadata.get(section_name)
            if isinstance(section, dict) and "K" in section:
                declared_strikes.append(float(section["K"]))
        if not declared_strikes:
            raise ValueError(f"Missing K declaration in {metadata_path}")
        for strike in declared_strikes:
            if not np.isclose(strike, 1.0):
                raise ValueError(f"Expected K=1 in {metadata_path}, found K={strike:g}")

def _load_compact_surface(directory: Path) -> dict[str, np.ndarray]:
    """
    Load a compact low-dimensional surface.

    Expected file:
        directory / "surface.npz"

    Required arrays:
        S_over_K
        tau_over_T
        V_over_K
        payoff_over_K
    """

    path = directory / "surface.npz"

    if not path.is_file():
        raise FileNotFoundError(path)

    with np.load(path) as archive:
        required = (
            "S_over_K",
            "tau_over_T",
            "V_over_K",
            "payoff_over_K",
        )

        missing = [
            name
            for name in required
            if name not in archive.files
        ]

        if missing:
            raise ValueError(
                f"Missing compact surface arrays in {directory}: {missing}"
            )

        surface = {
            name: np.asarray(
                archive[name],
                dtype=float,
            )
            for name in required
        }

    s = surface["S_over_K"]
    tau = surface["tau_over_T"]
    values = surface["V_over_K"]

    if values.shape != (tau.size, s.size):
        raise ValueError(
            f"Unexpected compact surface shape in {directory}: "
            f"{values.shape}; expected {(tau.size, s.size)}"
        )

    if not (
        np.isfinite(values).all()
        and np.isfinite(s).all()
        and np.isfinite(tau).all()
    ):
        raise ValueError(
            f"Non-finite compact surface data in {directory}"
        )

    return surface


def _load_hd_surface(directory: Path) -> dict[str, np.ndarray]:
    """
    Load a high-dimensional visualization surface and reshape it
    onto a rectangular (tau, S/K) grid.
    """

    path = directory / "surface.npz"

    if not path.is_file():
        raise FileNotFoundError(path)

    with np.load(path) as archive:
        required = (
            "S_over_K",
            "tau_over_T",
            "V_over_K",
        )

        missing = [
            name
            for name in required
            if name not in archive.files
        ]

        if missing:
            raise ValueError(
                f"Missing HD surface arrays in {directory}: {missing}"
            )

        s_flat = np.asarray(
            archive["S_over_K"],
            dtype=float,
        )

        tau_flat = np.asarray(
            archive["tau_over_T"],
            dtype=float,
        )

        values_flat = np.asarray(
            archive["V_over_K"],
            dtype=float,
        )

    s = np.unique(s_flat)
    tau = np.unique(tau_flat)

    if s_flat.size != s.size * tau.size:
        raise ValueError(
            f"HD surface is not rectangular: {directory}"
        )

    order = np.lexsort(
        (
            s_flat,
            tau_flat,
        )
    )

    values = values_flat[order].reshape(
        tau.size,
        s.size,
    )

    if not (
        np.isfinite(values).all()
        and np.isfinite(s).all()
        and np.isfinite(tau).all()
    ):
        raise ValueError(
            f"Non-finite HD surface data in {directory}"
        )

    return {
        "S_over_K": s,
        "tau_over_T": tau,
        "V_over_K": values,
    }


# ======================================================================
# Interpolation
# ======================================================================

def _time_slice(
    surface: dict[str, np.ndarray],
    tau_target: float,
) -> np.ndarray:
    """
    Obtain V(S) at a requested normalized time tau/T.

    If tau_target is not stored exactly, linearly interpolate between
    the two surrounding stored time slices.
    """

    tau = surface["tau_over_T"]
    values = surface["V_over_K"]

    if tau_target < tau[0] or tau_target > tau[-1]:
        raise ValueError(
            f"Requested tau {tau_target} outside stored domain "
            f"[{tau[0]}, {tau[-1]}]"
        )

    upper = int(
        np.searchsorted(
            tau,
            tau_target,
            side="left",
        )
    )

    if upper == 0:
        return values[0].copy()

    if upper < tau.size and np.isclose(
        tau[upper],
        tau_target,
    ):
        return values[upper].copy()

    if upper >= tau.size:
        return values[-1].copy()

    lower = upper - 1

    weight = (
        tau_target - tau[lower]
    ) / (
        tau[upper] - tau[lower]
    )

    return (
        (1.0 - weight) * values[lower]
        + weight * values[upper]
    )


def _on_common_s(
    surface: dict[str, np.ndarray],
    tau_target: float,
    s_common: np.ndarray,
) -> np.ndarray:
    """
    Evaluate one stored surface on the common S/K coordinates.
    """

    return np.interp(
        s_common,
        surface["S_over_K"],
        _time_slice(
            surface,
            tau_target,
        ),
    )


# ======================================================================
# Plot styling
# ======================================================================

def _style_axis(axis: plt.Axes) -> None:
    """
    Apply the plotting style used by the original comparison figures.
    """

    axis.grid(
        True,
        color="#D8D8D8",
        linewidth=0.6,
        alpha=0.8,
    )

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    axis.tick_params(
        axis="both",
        labelsize=11,
    )


# ======================================================================
# Fig01
# ======================================================================

def build_fig01(
    reference: dict[str, np.ndarray],
    arm_c: dict[str, np.ndarray],
    arm_d: dict[str, np.ndarray],
    fc: dict[str, np.ndarray],
    deepsets: dict[str, np.ndarray],
    *,
    tau_target: float = TAU_TARGET,
    figsize: tuple[float, float] = (10.0, 6.0),
    display_mode: str = "normalized",
) -> tuple[plt.Figure, plt.Axes]:
    """
    Build the revised Fig01 without writing files or opening a GUI.

    Curves:
        - High-Accuracy DIRK Reference
        - Vanilla soft-LCP (stored Arm C, seed 101 data)
        - ETC+FB (stored Arm D, seed 101 data)
        - FC-HVI (stored d50 data)
        - DeepSets-HVI (stored d50 data)

    Binomial Tree is intentionally omitted.
    """

    if display_mode not in DISPLAY_MODES:
        raise ValueError(f"Unknown display mode: {display_mode}")

    s = reference["S_over_K"]

    # ------------------------------------------------------------------
    # Validate approved common S/K domain
    # ------------------------------------------------------------------

    if s[0] > COMMON_S_MIN:
        raise ValueError(
            f"Reference surface starts at S/K={s[0]}, "
            f"but plotting requires {COMMON_S_MIN}."
        )

    if s[-1] < COMMON_S_MAX:
        raise ValueError(
            f"Reference surface ends at S/K={s[-1]}, "
            f"but plotting requires {COMMON_S_MAX}."
        )

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=figsize,
        constrained_layout=True,
    )

    # ------------------------------------------------------------------
    # High-accuracy DIRK reference
    # ------------------------------------------------------------------

    ax.plot(
        s,
        _on_common_s(
            reference,
            tau_target,
            s,
        ),
        color=COLORS["reference"],
        linewidth=2.6,
        label="High-Accuracy DIRK Reference",
        zorder=5,
    )

    # ------------------------------------------------------------------
    # Arm C
    # ------------------------------------------------------------------

    ax.plot(
        s,
        _on_common_s(
            arm_c,
            tau_target,
            s,
        ),
        color=COLORS["arm_c"],
        linewidth=2.0,
        label="Vanilla soft-LCP",
        zorder=4,
    )

    # ------------------------------------------------------------------
    # Arm D
    # ------------------------------------------------------------------

    ax.plot(
        s,
        _on_common_s(
            arm_d,
            tau_target,
            s,
        ),
        color=COLORS["arm_d"],
        linewidth=2.0,
        label="ETC+FB",
        zorder=4,
    )

    # ------------------------------------------------------------------
    # FC-HVI d20
    # ------------------------------------------------------------------

    ax.plot(
        s,
        _on_common_s(
            fc,
            tau_target,
            s,
        ),
        color=COLORS["fc"],
        linewidth=2.0,
        label="FC-HVI",
        zorder=3,
    )

    # ------------------------------------------------------------------
    # DeepSets-HVI d20
    # ------------------------------------------------------------------

    ax.plot(
        s,
        _on_common_s(
            deepsets,
            tau_target,
            s,
        ),
        color=COLORS["deepsets"],
        linewidth=2.0,
        label="DeepSets-HVI",
        zorder=3,
    )

    # ------------------------------------------------------------------
    # Axes
    # ------------------------------------------------------------------

    ax.set_xlim(
        COMMON_S_MIN,
        COMMON_S_MAX,
    )

    if display_mode == "normalized":
        xlabel = r"Normalized Stock Price $S/K$"
        ylabel = r"Normalized Option Value $V/K$"
    else:
        xlabel = r"Stock Price $S$"
        ylabel = r"Option Value $V$"

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)

    _style_axis(ax)

    # ------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------

    ax.legend(
        loc="lower left",
        fontsize=10,
        frameon=True,
        framealpha=0.95,
        edgecolor="#BBBBBB",
    )

    zoom_axis = ax.inset_axes(
        [0.53, 0.51, 0.45, 0.46],
        facecolor="white",
        zorder=10,
    )
    for line in ax.lines:
        zoom_axis.plot(
            line.get_xdata(),
            line.get_ydata(),
            color=line.get_color(),
            linewidth=max(1.35, 0.72 * line.get_linewidth()),
            label=line.get_label(),
            zorder=line.get_zorder(),
        )
    zoom_axis.set_xlim(0.95, 1.055)
    zoom_axis.set_ylim(0.15, 0.223)
    zoom_axis.grid(
        True,
        color="#D8D8D8",
        linewidth=0.5,
        alpha=0.8,
    )
    zoom_axis.tick_params(axis="both", labelsize=7.5)
    for spine in zoom_axis.spines.values():
        spine.set_color("#555555")
        spine.set_linewidth(1.0)

    zoom_rectangle = Rectangle(
        (0.95, 0.15),
        0.105,
        0.073,
        facecolor=(0.90, 0.12, 0.12, 0.14),
        edgecolor="#C62828",
        linewidth=1.6,
        zorder=6,
    )
    ax.add_patch(zoom_rectangle)

    zoom_arrow = ConnectionPatch(
        xyA=(1.055, 0.223),
        coordsA=ax.transData,
        xyB=(0.16, 0.0),
        coordsB=zoom_axis.transAxes,
        arrowstyle="-|>",
        mutation_scale=15,
        shrinkA=2,
        shrinkB=3,
        color="#C62828",
        linewidth=1.8,
        zorder=11,
    )
    fig.add_artist(zoom_arrow)

    return fig, ax


def save_current_view(
    fig: plt.Figure,
    output_dir: Path,
    *,
    stem: str = "fig01_price_slice_tau05_selected",
) -> list[Path]:
    """Save the figure exactly as it currently appears in the GUI."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    fig.savefig(
        png_path,
        dpi=360,
        bbox_inches="tight",
    )

    return [pdf_path, png_path]


def describe_current_view(fig: plt.Figure, axis: plt.Axes) -> str:
    """Return reproducible geometry and zoom information for the current view."""

    canvas_width, canvas_height = fig.canvas.get_width_height()
    figure_width, figure_height = fig.get_size_inches()
    left, bottom, width, height = axis.get_position().bounds
    x_min, x_max = axis.get_xlim()
    y_min, y_max = axis.get_ylim()
    return "\n".join(
        (
            f"Canvas pixels: {canvas_width} x {canvas_height}",
            f"Figure inches: {figure_width:.3f} x {figure_height:.3f} at {fig.dpi:g} dpi",
            f"Axes position: left={left:.4f}, bottom={bottom:.4f}, width={width:.4f}, height={height:.4f}",
            f"Axes x-range: {x_min:.6g} to {x_max:.6g}",
            f"Axes y-range: {y_min:.6g} to {y_max:.6g}",
        )
    )


def connect_interactive_shortcuts(
    fig: plt.Figure,
    axis: plt.Axes,
    output_dir: Path,
    *,
    stem: str = "fig01_price_slice_tau05_selected",
) -> int:
    """Bind P to report the view and S to save the current GUI state."""

    def on_key(event: object) -> None:
        key = str(getattr(event, "key", "")).lower()
        if key == "p":
            print()
            print(describe_current_view(fig, axis))
            print()
        elif key == "s":
            paths = save_current_view(fig, output_dir, stem=stem)
            print()
            print("Saved current view:")
            for path in paths:
                print(f"  {path}")
            print(describe_current_view(fig, axis))
            print()

    return fig.canvas.mpl_connect("key_press_event", on_key)


def generate_fig01(
    reference: dict[str, np.ndarray],
    arm_c: dict[str, np.ndarray],
    arm_d: dict[str, np.ndarray],
    fc: dict[str, np.ndarray],
    deepsets: dict[str, np.ndarray],
    output_dir: Path,
    *,
    tau_target: float = TAU_TARGET,
    figsize: tuple[float, float] = (10.0, 6.0),
    display_mode: str = "normalized",
    interactive: bool = True,
) -> list[Path]:
    """Build the plot, save its initial view, and optionally open the GUI."""

    fig, axis = build_fig01(
        reference,
        arm_c,
        arm_d,
        fc,
        deepsets,
        tau_target=tau_target,
        figsize=figsize,
        display_mode=display_mode,
    )
    stem = (
        "fig01_price_slice_tau05_K1"
        if display_mode == "k1"
        else "fig01_price_slice_tau05_selected"
    )
    connect_interactive_shortcuts(fig, axis, output_dir, stem=stem)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    paths = save_current_view(fig, output_dir, stem=stem)

    print()
    print("Generated:")
    print(f"  PDF: {paths[0]}")
    print(f"  PNG: {paths[1]}")
    print()
    print(
        f"Figure size: "
        f"{figsize[0]:g} × {figsize[1]:g} inches"
    )

    # ------------------------------------------------------------------
    # Interactive Matplotlib window
    # ------------------------------------------------------------------

    if interactive:
        print(
            "Opening interactive Matplotlib window..."
        )
        print(
            "Use the toolbar to zoom/pan, or resize "
            "the macOS window manually."
        )
        print("Press P to print the current size and zoom range.")
        print("Press S to save the currently selected view as PNG and PDF.")
        print()

        plt.show(
            block=True,
        )

    else:
        plt.close(fig)

    return paths


# ======================================================================
# Main
# ======================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--hd-root",
        type=Path,
        default=DEFAULT_HD_ROOT,
        help=(
            "Root directory containing HD results, "
            "including visualization/d50/FC_HVI and "
            "visualization/d50/DEEPSETS_HVI."
        ),
    )

    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE,
                        help="Stored DIRK reference export.")
    parser.add_argument("--arm-c", type=Path, default=DEFAULT_ARM_C,
                        help="Stored Arm C poster-case export.")
    parser.add_argument("--arm-d", type=Path, default=DEFAULT_ARM_D,
                        help="Stored Arm D poster-case export.")

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Directory where the PDF and PNG will be written."
        ),
    )

    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        metavar=(
            "WIDTH",
            "HEIGHT",
        ),
        default=(
            10.0,
            6.0,
        ),
        help=(
            "Figure size in inches. "
            "Default: --figsize 10 6. "
            "Try --figsize 12 7 for a larger layout."
        ),
    )

    parser.add_argument(
        "--display-mode",
        choices=DISPLAY_MODES,
        default="normalized",
        help=(
            "Use normalized S/K and V/K labels, or absolute S and V labels "
            "for the verified K=1 matched contract."
        ),
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help=(
            "Save the figure without opening "
            "the interactive Matplotlib window."
        ),
    )

    args = parser.parse_args()

    hd_root = args.hd_root.resolve()
    output_dir = args.output_dir.resolve()

    # ------------------------------------------------------------------
    # Required source paths
    # ------------------------------------------------------------------

    reference_dir = args.reference.resolve()
    arm_c_dir = args.arm_c.resolve()
    arm_d_dir = args.arm_d.resolve()

    print()
    print("=" * 72)
    print("R3 price slice")
    print("=" * 72)
    print(f"Regime:      {REGIME}")
    print(f"tau/T:       {TAU_TARGET}")
    print(f"Display mode:{args.display_mode}")
    print(f"Reference:   {reference_dir}")
    print(f"Arm C:       {arm_c_dir}")
    print(f"Arm D:       {arm_d_dir}")
    print(f"HD root:     {hd_root}")
    print(f"Output:      {output_dir}")
    print("=" * 72)
    print()

    fc_dir, deepsets_dir = high_dimensional_surface_dirs(hd_root)
    validate_k1_sources(
        [reference_dir, arm_c_dir, arm_d_dir, fc_dir, deepsets_dir]
    )

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------

    print("Loading stored surfaces...")

    reference = _load_compact_surface(
        reference_dir
    )

    arm_c = _load_compact_surface(
        arm_c_dir
    )

    arm_d = _load_compact_surface(
        arm_d_dir
    )

    fc = _load_hd_surface(
        fc_dir
    )

    deepsets = _load_hd_surface(
        deepsets_dir
    )

    # ------------------------------------------------------------------
    # Validate low-dimensional coordinates
    # ------------------------------------------------------------------

    if not np.array_equal(
        reference["S_over_K"],
        arm_c["S_over_K"],
    ):
        raise ValueError(
            "Arm C S/K coordinates do not match "
            "the stored reference."
        )

    if not np.array_equal(
        reference["S_over_K"],
        arm_d["S_over_K"],
    ):
        raise ValueError(
            "Arm D S/K coordinates do not match "
            "the stored reference."
        )

    print("Surface loading and validation passed.")

    # ------------------------------------------------------------------
    # Generate figure
    # ------------------------------------------------------------------

    generate_fig01(
        reference=reference,
        arm_c=arm_c,
        arm_d=arm_d,
        fc=fc,
        deepsets=deepsets,
        output_dir=output_dir,
        tau_target=TAU_TARGET,
        figsize=(
            args.figsize[0],
            args.figsize[1],
        ),
        display_mode=args.display_mode,
        interactive=not args.no_show,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
