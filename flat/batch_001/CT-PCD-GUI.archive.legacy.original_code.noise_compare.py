import csv
import glob
import os
import queue
import threading

import matplotlib.pyplot as plt
import numpy as np
from pydicom import dcmread
from skimage.metrics import structural_similarity as ssim

# =====================================================================
# Configuration
# =====================================================================

TRUTH_ROOT = r"U:\PUBLIC\DICOM-CTPD\PCD_DICOMCTPD_Generator\Quarter_dose_CTPD_File_CLI"

TEST_ROOT = (
    r"U:\PUBLIC\DICOM-CTPD\PCD_DICOMCTPD_Generator" r"\Full_dose_CTPD_File_noise_output"
)

RESULTS_CSV = os.path.abspath("projection_comparison_metrics.csv")

# All quantitative metrics use the full-resolution projection data.
# Only the displayed images and SSIM inputs are downsampled.
DISPLAY_MAX_DIMENSION = 1024

# There is generally no benefit in asking Matplotlib to check the queue
# every 16 ms when DICOM and metric calculations take much longer.
TIMER_INTERVAL_MS = 50

# For comparable SSIM values across projections, set this to one fixed,
# physically meaningful range in the native projection units.
#
# Example only:
# FIXED_SSIM_DATA_RANGE = 65535.0
#
# When set to None, each projection pair uses its own combined range.
# That is acceptable for a secondary visual check, but the SSIM scores
# are less directly comparable across projection angles.
FIXED_SSIM_DATA_RANGE = None


# =====================================================================
# DICOM and metric functions
# =====================================================================


def get_projection(filename):
    """
    Read one 2-D DICOM projection as float32.

    RescaleSlope and RescaleIntercept are applied when present so the
    comparison is performed in the represented projection units.
    """
    ds = dcmread(filename, force=True)

    projection = np.array(
        ds.pixel_array,
        dtype=np.float32,
        copy=True,
    )
    projection = np.squeeze(projection)

    if projection.ndim != 2:
        raise ValueError(
            f"Expected a 2-D projection, but {filename!r} "
            f"has shape {projection.shape}."
        )

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))

    if slope != 1.0:
        projection *= slope

    if intercept != 0.0:
        projection += intercept

    if not np.isfinite(projection).all():
        raise ValueError(f"{filename!r} contains NaN or infinite values.")

    return projection


def get_display_step(shape):
    """
    Choose an integer stride that limits the displayed image size.

    This affects only display and SSIM. Native-unit metrics still use
    every projection sample.
    """
    largest_dimension = max(shape)

    return max(
        1,
        int(np.ceil(largest_dimension / DISPLAY_MAX_DIMENSION)),
    )


def get_display_limits(reference, comparison):
    """
    Calculate shared robust grayscale limits for display only.

    The 0.5th and 99.5th percentiles prevent isolated extreme values
    from destroying the displayed contrast.
    """
    vmin = min(
        float(np.percentile(reference, 0.5)),
        float(np.percentile(comparison, 0.5)),
    )

    vmax = max(
        float(np.percentile(reference, 99.5)),
        float(np.percentile(comparison, 99.5)),
    )

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError("Could not determine finite display limits.")

    if vmax <= vmin:
        vmin = min(
            float(reference.min()),
            float(comparison.min()),
        )
        vmax = max(
            float(reference.max()),
            float(comparison.max()),
        )

    if vmax <= vmin:
        vmax = vmin + 1.0

    return vmin, vmax


def calculate_ssim(reference, comparison, data_range):
    """
    Calculate SSIM as a secondary structural metric.

    The input arrays are downsampled display arrays. SSIM is not used
    as the primary test of stochastic noise equivalence.
    """
    if not np.isfinite(data_range) or data_range <= 0:
        return np.nan

    minimum_dimension = min(reference.shape)

    # skimage requires an odd window size no larger than the image.
    window_size = min(7, minimum_dimension)

    if window_size % 2 == 0:
        window_size -= 1

    if window_size < 3:
        return np.nan

    return float(
        ssim(
            reference,
            comparison,
            data_range=float(data_range),
            win_size=window_size,
        )  # type: ignore
    )


def calculate_frame(
    index,
    truth_filename,
    test_filename,
):
    """
    Load and compare one projection pair.

    Bias, MAE, MSE, RMSE, and difference SD are calculated from the
    full-resolution native-unit projection values.
    """
    proj_truth = get_projection(truth_filename)
    proj_test = get_projection(test_filename)

    if proj_truth.shape != proj_test.shape:
        raise ValueError(
            f"Shape mismatch for projection {index}:\n"
            f"Truth: {truth_filename} -> {proj_truth.shape}\n"
            f"Test:  {test_filename} -> {proj_test.shape}"
        )

    # Positive values mean the GUI projection is larger.
    difference = proj_test - proj_truth

    pixel_count = difference.size

    # These sums are retained so the worker can calculate exact running
    # statistics over all pixels processed so far.
    sum_difference = float(
        np.sum(
            difference,
            dtype=np.float64,
        )
    )

    sum_absolute = float(
        np.sum(
            np.abs(difference),
            dtype=np.float64,
        )
    )

    # einsum accumulates in float64 without first constructing a
    # full-resolution float64 squared-difference image.
    sum_squared = float(
        np.einsum(
            "ij,ij->",
            difference,
            difference,
            dtype=np.float64,
        )
    )

    bias = sum_difference / pixel_count
    mae = sum_absolute / pixel_count
    mse = sum_squared / pixel_count
    rmse = float(np.sqrt(max(mse, 0.0)))

    difference_sd = float(
        np.std(
            difference,
            dtype=np.float64,
            ddof=1,
        )
    )

    # Downsample only after all full-resolution metrics are calculated.
    display_step = get_display_step(proj_truth.shape)

    truth_display = np.ascontiguousarray(
        proj_truth[
            ::display_step,
            ::display_step,
        ]
    )

    test_display = np.ascontiguousarray(
        proj_test[
            ::display_step,
            ::display_step,
        ]
    )

    difference_display = np.ascontiguousarray(
        difference[
            ::display_step,
            ::display_step,
        ]
    )

    display_vmin, display_vmax = get_display_limits(
        truth_display,
        test_display,
    )

    # Use symmetric difference limits so zero is always at the center
    # of the difference colormap.
    difference_limit = float(
        np.percentile(
            np.abs(difference_display),
            99.5,
        )
    )

    if not np.isfinite(difference_limit) or difference_limit <= 0:
        difference_limit = float(np.max(np.abs(difference_display)))

    if not np.isfinite(difference_limit) or difference_limit <= 0:
        difference_limit = 1.0

    pair_min = min(
        float(proj_truth.min()),
        float(proj_test.min()),
    )

    pair_max = max(
        float(proj_truth.max()),
        float(proj_test.max()),
    )

    pair_data_range = pair_max - pair_min

    if FIXED_SSIM_DATA_RANGE is None:
        ssim_data_range = pair_data_range
    else:
        ssim_data_range = float(FIXED_SSIM_DATA_RANGE)

    ssim_value = calculate_ssim(
        truth_display,
        test_display,
        ssim_data_range,
    )

    return {
        "index": index,
        "truth_filename": truth_filename,
        "test_filename": test_filename,
        "truth_display": truth_display,
        "test_display": test_display,
        "difference_display": difference_display,
        "display_vmin": display_vmin,
        "display_vmax": display_vmax,
        "difference_limit": difference_limit,
        "pixel_count": pixel_count,
        "sum_difference": sum_difference,
        "sum_absolute": sum_absolute,
        "sum_squared": sum_squared,
        "bias": bias,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "difference_sd": difference_sd,
        "ssim": ssim_value,
        "ssim_data_range": ssim_data_range,
    }


def summarize_running(
    total_pixel_count,
    total_sum_difference,
    total_sum_absolute,
    total_sum_squared,
):
    """
    Calculate metrics over all projection pixels processed so far.

    This is preferable to averaging per-projection RMSE values because:

        global RMSE = sqrt(global mean squared error)

    rather than:

        mean(per-projection RMSE)
    """
    if total_pixel_count <= 0:
        raise ValueError("total_pixel_count must be positive.")

    bias = total_sum_difference / total_pixel_count

    mae = total_sum_absolute / total_pixel_count

    mse = total_sum_squared / total_pixel_count

    rmse = float(np.sqrt(max(mse, 0.0)))

    if total_pixel_count > 1:
        centered_sum_squared = (
            total_sum_squared
            - (total_sum_difference * total_sum_difference) / total_pixel_count
        )

        difference_variance = max(
            centered_sum_squared / (total_pixel_count - 1),
            0.0,
        )

        difference_sd = float(np.sqrt(difference_variance))
    else:
        difference_sd = 0.0

    return {
        "bias": float(bias),
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "difference_sd": float(difference_sd),
    }


def write_results_csv(records):
    """Write one row of metrics for each projection pair."""
    fields = [
        "index",
        "truth_file",
        "test_file",
        "pixel_count",
        "bias",
        "mae",
        "mse",
        "rmse",
        "difference_sd",
        "ssim",
        "ssim_data_range",
    ]

    with open(
        RESULTS_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(records)


# =====================================================================
# Locate and pair DICOM files
# =====================================================================

qd_truth = sorted(
    glob.glob(
        os.path.join(
            TRUTH_ROOT,
            "**",
            "*.dcm",
        ),
        recursive=True,
    )
)

qd_test = sorted(
    glob.glob(
        os.path.join(
            TEST_ROOT,
            "**",
            "*.dcm",
        ),
        recursive=True,
    )
)

if not qd_truth:
    raise FileNotFoundError(f"No DICOM files were found under " f"{TRUTH_ROOT!r}.")

if not qd_test:
    raise FileNotFoundError(f"No DICOM files were found under " f"{TEST_ROOT!r}.")

if len(qd_truth) != len(qd_test):
    raise ValueError(
        f"Found {len(qd_truth)} truth files and "
        f"{len(qd_test)} test files. "
        f"Correct the file pairing before running "
        f"the comparison."
    )

pairs = list(
    zip(
        qd_truth,
        qd_test,
    )
)

pair_count = len(pairs)

# Sorting does not prove that two files correspond. This warning catches
# one common pairing problem while still allowing intentionally different
# output filenames.
filename_mismatches = sum(
    os.path.basename(truth_filename) != os.path.basename(test_filename)
    for truth_filename, test_filename in pairs
)

if filename_mismatches:
    print(
        f"Warning: {filename_mismatches} sorted pairs "
        f"have different basenames. Verify that "
        f"lexicographic sorting pairs corresponding "
        f"projections."
    )


# =====================================================================
# Create the figure
# =====================================================================

initial_shape = get_projection(pairs[0][0]).shape

initial_step = get_display_step(initial_shape)

initial_display_shape = (
    (initial_shape[0] + initial_step - 1) // initial_step,
    (initial_shape[1] + initial_step - 1) // initial_step,
)

placeholder = np.zeros(
    initial_display_shape,
    dtype=np.float32,
)

fig, axes = plt.subplots(
    nrows=1,
    ncols=3,
    figsize=(14, 5.5),
    sharex=True,
    sharey=True,
)

fig.subplots_adjust(
    left=0.04,
    right=0.98,
    bottom=0.23,
    top=0.84,
    wspace=0.08,
)

im_truth = axes[0].imshow(
    placeholder,
    cmap="gray",
    vmin=0,
    vmax=1,
    interpolation="nearest",
)
axes[0].set_title("Original Quarter Dose")

im_test = axes[1].imshow(
    placeholder,
    cmap="gray",
    vmin=0,
    vmax=1,
    interpolation="nearest",
)
axes[1].set_title("GUI Quarter Dose")

im_difference = axes[2].imshow(
    placeholder,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    interpolation="nearest",
)
axes[2].set_title("Difference: GUI - Original")

for axis in axes:
    axis.set_axis_off()

status_text = fig.suptitle(f"Preparing {pair_count} projection pairs")

metrics_text = fig.text(
    0.5,
    0.05,
    ("Bias: N/A | MAE: N/A | RMSE: N/A | " "Difference SD: N/A | SSIM*: N/A"),
    ha="center",
    va="bottom",
    family="monospace",
    fontsize=9,
)


# =====================================================================
# Worker and GUI update loop
# =====================================================================

# A queue size of one prevents multiple large display frames from
# accumulating in memory.
frame_queue = queue.Queue(maxsize=1)

stop_event = threading.Event()


def put_message(message):
    """
    Put a message into the GUI queue unless the figure was closed.
    """
    while not stop_event.is_set():
        try:
            frame_queue.put(
                message,
                timeout=0.1,
            )
            return True

        except queue.Full:
            continue

    return False


def calculation_worker():
    """
    Calculate projection metrics outside Matplotlib's GUI thread.
    """
    total_pixel_count = 0
    total_sum_difference = 0.0
    total_sum_absolute = 0.0
    total_sum_squared = 0.0

    running_ssim = 0.0
    ssim_count = 0

    records = []

    try:
        for index, (
            truth_filename,
            test_filename,
        ) in enumerate(
            pairs,
            start=1,
        ):
            if stop_event.is_set():
                return

            frame = calculate_frame(
                index,
                truth_filename,
                test_filename,
            )

            total_pixel_count += frame["pixel_count"]

            total_sum_difference += frame["sum_difference"]

            total_sum_absolute += frame["sum_absolute"]

            total_sum_squared += frame["sum_squared"]

            running = summarize_running(
                total_pixel_count,
                total_sum_difference,
                total_sum_absolute,
                total_sum_squared,
            )

            current_ssim = frame["ssim"]

            if np.isfinite(current_ssim):
                ssim_count += 1

                running_ssim += (current_ssim - running_ssim) / ssim_count

            running["ssim"] = float(running_ssim) if ssim_count else np.nan

            frame["running"] = running

            records.append(
                {
                    "index": index,
                    "truth_file": truth_filename,
                    "test_file": test_filename,
                    "pixel_count": frame["pixel_count"],
                    "bias": frame["bias"],
                    "mae": frame["mae"],
                    "mse": frame["mse"],
                    "rmse": frame["rmse"],
                    "difference_sd": frame["difference_sd"],
                    "ssim": frame["ssim"],
                    "ssim_data_range": frame["ssim_data_range"],
                }
            )

            if not put_message(
                (
                    "frame",
                    frame,
                )
            ):
                return

        csv_error = None

        try:
            write_results_csv(records)
        except OSError as error:
            csv_error = str(error)

        final_summary = summarize_running(
            total_pixel_count,
            total_sum_difference,
            total_sum_absolute,
            total_sum_squared,
        )

        final_summary["ssim"] = float(running_ssim) if ssim_count else np.nan

        put_message(
            (
                "done",
                {
                    "count": pair_count,
                    "summary": final_summary,
                    "csv": RESULTS_CSV,
                    "csv_error": csv_error,
                },
            )
        )

    except Exception as error:
        put_message(
            (
                "error",
                error,
            )
        )


def format_metric(value):
    """
    Format a metric without converting small values to an apparent 0.00.
    """
    if value is None or not np.isfinite(value):
        return "N/A"

    return f"{value:.6g}"


def update_figure():
    """
    Apply calculated results to Matplotlib on the GUI thread.
    """
    try:
        message_type, payload = frame_queue.get_nowait()

    except queue.Empty:
        return

    if message_type == "frame":
        frame = payload
        running = frame["running"]

        im_truth.set_data(frame["truth_display"])
        im_truth.set_clim(
            frame["display_vmin"],
            frame["display_vmax"],
        )

        im_test.set_data(frame["test_display"])
        im_test.set_clim(
            frame["display_vmin"],
            frame["display_vmax"],
        )

        im_difference.set_data(frame["difference_display"])
        im_difference.set_clim(
            -frame["difference_limit"],
            frame["difference_limit"],
        )

        metrics_text.set_text(
            "Current / running over all processed pixels\n"
            f"Bias: {format_metric(frame['bias'])} / "
            f"{format_metric(running['bias'])}    "
            f"MAE: {format_metric(frame['mae'])} / "
            f"{format_metric(running['mae'])}    "
            f"RMSE: {format_metric(frame['rmse'])} / "
            f"{format_metric(running['rmse'])}\n"
            f"Difference SD: "
            f"{format_metric(frame['difference_sd'])} / "
            f"{format_metric(running['difference_sd'])}    "
            f"SSIM*: {format_metric(frame['ssim'])} / "
            f"{format_metric(running['ssim'])}"
        )

        status_text.set_text(
            f"Projection {frame['index']} "
            f"of {pair_count}: "
            f"{os.path.basename(frame['truth_filename'])}"
        )

        fig.canvas.draw_idle()

    elif message_type == "error":
        stop_event.set()
        timer.stop()

        status_text.set_text(f"Processing stopped: {payload}")

        fig.canvas.draw_idle()

        print(f"Processing error: {payload}")

    elif message_type == "done":
        timer.stop()

        summary = payload["summary"]

        status_text.set_text(
            f"Complete: processed " f"{payload['count']} projection pairs"
        )

        metrics_text.set_text(
            "Final native-unit summary over all compared pixels\n"
            f"Bias: {format_metric(summary['bias'])}    "
            f"MAE: {format_metric(summary['mae'])}    "
            f"RMSE: {format_metric(summary['rmse'])}    "
            f"Difference SD: "
            f"{format_metric(summary['difference_sd'])}    "
            f"Mean SSIM*: "
            f"{format_metric(summary['ssim'])}"
        )

        fig.canvas.draw_idle()

        if payload["csv_error"] is None:
            print("Per-projection metrics written to: " f"{payload['csv']}")
        else:
            print(
                "Metrics were calculated, but the CSV "
                "could not be written: "
                f"{payload['csv_error']}"
            )


def on_figure_closed(event):
    """Signal the worker to stop when the figure is closed."""
    stop_event.set()
    timer.stop()


timer = fig.canvas.new_timer(interval=TIMER_INTERVAL_MS)

timer.add_callback(update_figure)

fig.canvas.mpl_connect(
    "close_event",
    on_figure_closed,
)

worker = threading.Thread(
    target=calculation_worker,
    daemon=True,
)

worker.start()
timer.start()

# Explicit blocking is useful when this is run from an IDE with
# interactive plotting enabled.
plt.show(block=True)

stop_event.set()
