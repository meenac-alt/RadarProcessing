

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


# ============================================================
# CONFIG
# ============================================================

FILES = {
    "master": r"y_door_in\master_0000_data.bin",
    "slave1": r"y_door_in\slave1_0000_data.bin",
    "slave2": r"y_door_in\slave2_0000_data.bin",
    "slave3": r"y_door_in\slave3_0000_data.bin",
}
# FILES = {
#     "master": r"inbox_small_range\master_0000_data.bin",
#     "slave1": r"inbox_small_range\slave1_0000_data.bin",
#     "slave2": r"inbox_small_range\slave2_0000_data.bin",
#     "slave3": r"inbox_small_range\slave3_0000_data.bin",
# }


# ------------------------------------------------------------
# TI radar parameters
# ------------------------------------------------------------

START_FREQ_HZ = 77e9

# 15.0148 MHz/us
# = 15.0148e12 Hz/s
SLOPE_HZ_PER_S = 79.0327e12

NUM_ADC_SAMPLES = 256
NUM_RX = 4

# IMPORTANT:
# This must correspond to the actual TI digOutSampleRate.
# 10000 ksps = 10 MHz
FS_HZ = 8e6

C0 = 299792458.0

# ------------------------------------------------------------
# FFT
# ------------------------------------------------------------

NFFT = NUM_ADC_SAMPLES

# ------------------------------------------------------------
# Display
# ------------------------------------------------------------

PLOT_DB_FLOOR = -80.0

# ------------------------------------------------------------
# Peak detection
# ------------------------------------------------------------

PEAK_THRESHOLD_DB = -55.0

# Minimum separation between detected peaks.
# This value is in FFT bins.
MIN_PEAK_DISTANCE = 2


# ============================================================
# LOAD RAW TI DCA1000 DATA
# ============================================================

def load_dca1000_iq(
    file_path,
    num_rx,
    num_adc_samples
):
    """
    Load TI DCA1000-style complex ADC capture.

    Assumed raw organization:
        I0, Q0, I1, Q1, ...

    Returns
    -------
    adc_cube : complex ndarray
        Shape:
            (num_chirps, num_rx, num_adc_samples)
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"\nFile not found:\n{file_path}"
        )

    # --------------------------------------------------------
    # Read raw signed 16-bit ADC values
    # --------------------------------------------------------

    raw = np.fromfile(
        file_path,
        dtype=np.int16
    )

    file_size = os.path.getsize(file_path)

    print(f"File: {file_path}")
    print(f"File size: {file_size} bytes")
    print(f"Raw int16 values: {raw.size}")

    # --------------------------------------------------------
    # Check for odd number of int16 values
    # --------------------------------------------------------

    if raw.size % 2 != 0:
        print(
            "WARNING: Odd number of int16 values. "
            "Dropping the last value."
        )
        raw = raw[:-1]

    # --------------------------------------------------------
    # Reconstruct complex I/Q
    # --------------------------------------------------------

    I = raw[0::2].astype(np.float32)
    Q = raw[1::2].astype(np.float32)

    complex_data = I + 1j * Q

    print(
        f"Complex samples: {complex_data.size}"
    )

    # --------------------------------------------------------
    # Samples per chirp
    # --------------------------------------------------------

    samples_per_chirp = (
        num_rx * num_adc_samples
    )

    print(
        f"Samples/chirp: {samples_per_chirp}"
    )

    # --------------------------------------------------------
    # Check reshape
    # --------------------------------------------------------

    remainder = (
        complex_data.size
        % samples_per_chirp
    )

    if remainder != 0:
        raise ValueError(
            "\nCannot reshape the raw capture.\n"
            f"Complex samples = {complex_data.size}\n"
            f"Samples/chirp   = {samples_per_chirp}\n"
            f"Remainder       = {remainder}\n\n"
            "Possible causes:\n"
            "1. Incorrect number of RX channels.\n"
            "2. Incorrect I/Q interleaving.\n"
            "3. Cascade data has a different layout.\n"
            "4. The capture contains additional metadata.\n"
        )

    # --------------------------------------------------------
    # Number of chirps
    # --------------------------------------------------------

    num_chirps = (
        complex_data.size
        // samples_per_chirp
    )

    print(
        f"Number of chirps: {num_chirps}"
    )

    # --------------------------------------------------------
    # Reshape
    # --------------------------------------------------------

    # adc_cube = complex_data.reshape(
    #     num_chirps,
    #     num_rx,
    #     num_adc_samples
    # )
    adc_cube = complex_data.reshape(
    num_chirps,
    num_adc_samples,
    num_rx
    ).transpose(0, 2, 1)  # Rearrange to (chirps, RX, ADC samples)

    print(
        f"ADC cube shape: {adc_cube.shape}"
    )

    return adc_cube


# ============================================================
# RANGE FFT
# ============================================================

def compute_range_fft(
    adc_cube,
    fs_hz,
    slope_hz_per_s,
    nfft
):
    """
    Compute range FFT along the ADC-sample dimension.

    Input
    -----
    adc_cube:
        (chirps, RX, ADC samples)

    Output
    ------
    range_fft:
        (chirps, RX, range bins)

    range_axis_m:
        range corresponding to each FFT bin
    """

    num_adc_samples = adc_cube.shape[-1]

    # --------------------------------------------------------
    # Hann window along fast-time
    # --------------------------------------------------------

    window = np.hanning(
        num_adc_samples
    )

    adc_windowed = (
        adc_cube
        * window[None, None, :]
    )

    # --------------------------------------------------------
    # Range FFT
    # --------------------------------------------------------

    range_fft_full = np.fft.fft(
        adc_windowed,
        n=nfft,
        axis=-1
    )

    # # --------------------------------------------------------
    # # Keep positive frequencies only
    # # --------------------------------------------------------

    # num_range_bins = nfft // 2
    # # num_range_bins = nfft 
    # print(
    #     f"Range FFT shape (full) : {range_fft_full.shape}"
    # )
    # range_fft = (
    #     range_fft_full[
    #         ...,
    #         :num_range_bins
    #     ]
    # )
    # print(
    #     f"Range FFT shape (half) : {range_fft.shape}"
    # )

    # # --------------------------------------------------------
    # # Frequency axis
    # # --------------------------------------------------------

    # freq_axis = np.fft.fftfreq(
    #     nfft,
    #     d=1.0 / fs_hz
    # )[:num_range_bins]

    # # --------------------------------------------------------
    # # Beat frequency -> range
    # #
    # # R = c * fb / (2*S)
    # # --------------------------------------------------------

    # range_axis_m = (
    #     C0
    #     * freq_axis
    #     / (2.0 * slope_hz_per_s)
    # )

    # print(
    #     f"Range axis (m) shape : {range_axis_m.shape}"
    # )

    # return range_fft, range_axis_m

    # --------------------------------------------------------
    # Keep ALL FFT bins
    # --------------------------------------------------------

    num_range_bins = nfft

    range_fft = range_fft_full

    print(
        f"Range FFT shape : {range_fft.shape}"
    )

    # --------------------------------------------------------
    # Beat-frequency axis
    #
    # Treat bins as:
    #   bin 0   -> 0
    #   bin 1   -> Fs/NFFT
    #   ...
    #   bin 255 -> 255*Fs/NFFT
    # --------------------------------------------------------

    freq_axis = (
        np.arange(nfft)
        * fs_hz
        / nfft
    )

    # --------------------------------------------------------
    # Beat frequency -> range
    #
    # R = c * fb / (2*S)
    # --------------------------------------------------------

    range_axis_m = (
        C0
        * freq_axis
        / (2.0 * slope_hz_per_s)
    )

    print(
        f"Range axis (m) shape : {range_axis_m.shape}"
    )

    return range_fft, range_axis_m


# ============================================================
# COMBINE CHIRPS + RX FOR ONE CHIP
# ============================================================

def combine_chip_range_power(range_fft):
    """
    Non-coherently combine all chirps and RX channels.

    P(R) = sum |X(chirp, RX, R)|^2
    """

    range_power = (
        np.abs(range_fft) ** 2
    )

    combined_power = np.sum(
        range_power,
        axis=(0, 1)
    )

    return combined_power


# ============================================================
# POWER -> NORMALIZED dB
# ============================================================

def power_to_db(power):
    """
    Convert power to normalized dB.
    Strongest peak becomes 0 dB.
    """

    db = 10.0 * np.log10(
        power + 1e-12
    )

    db -= np.max(db)

    db = np.maximum(
        db,
        PLOT_DB_FLOOR
    )

    return db


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TI CASCADE RANGE FFT PROCESSING")
    print("=" * 70)

    print("\nRadar parameters:")
    print(f"Start frequency : {START_FREQ_HZ / 1e9:.3f} GHz")
    print(
        f"Chirp slope     : "
        f"{SLOPE_HZ_PER_S / 1e12:.4f} MHz/us"
    )
    print(
        f"ADC sample rate : "
        f"{FS_HZ / 1e6:.3f} MSPS"
    )
    print(
        f"ADC samples     : {NUM_ADC_SAMPLES}"
    )
    print(
        f"RX channels     : {NUM_RX}"
    )
    print(
        f"NFFT            : {NFFT}"
    )

    # ========================================================
    # STORE RESULTS
    # ========================================================

    all_range_fft = {}
    all_range_power = {}
    all_range_db = {}

    range_axis_m = None

    # ========================================================
    # PROCESS EACH CASCADE CHIP
    # ========================================================

    for chip_name, file_path in FILES.items():

        print("\n")
        print("=" * 70)
        print(f"PROCESSING {chip_name.upper()}")
        print("=" * 70)

        # ----------------------------------------------------
        # Load ADC data
        # ----------------------------------------------------

        adc_cube = load_dca1000_iq(
            file_path=file_path,
            num_rx=NUM_RX,
            num_adc_samples=NUM_ADC_SAMPLES
        )

        print(
            f"\n{chip_name} ADC cube:"
        )
        print(
            f"    shape = {adc_cube.shape}"
        )
        print(
            "    format = "
            "(chirp, RX, ADC sample)"
        )

        # ----------------------------------------------------
        # Range FFT
        # ----------------------------------------------------

        range_fft, current_range_axis_m = (
            compute_range_fft(
                adc_cube=adc_cube,
                fs_hz=FS_HZ,
                slope_hz_per_s=SLOPE_HZ_PER_S,
                nfft=NFFT
            )
        )

        # ----------------------------------------------------
        # Save FFT
        # ----------------------------------------------------

        all_range_fft[chip_name] = (
            range_fft
        )

        # ----------------------------------------------------
        # Check range axis consistency
        # ----------------------------------------------------

        if range_axis_m is None:

            range_axis_m = (
                current_range_axis_m
            )

        else:

            if not np.allclose(
                range_axis_m,
                current_range_axis_m
            ):
                raise ValueError(
                    "Range axes are different "
                    "between cascade chips."
                )

        # ----------------------------------------------------
        # Combine all chirps + RX
        # ----------------------------------------------------

        combined_power = (
            combine_chip_range_power(
                range_fft
            )
        )

        all_range_power[chip_name] = (
            combined_power
        )

        # ----------------------------------------------------
        # Convert to dB
        # ----------------------------------------------------

        combined_db = power_to_db(
            combined_power
        )

        all_range_db[chip_name] = (
            combined_db
        )

        # ----------------------------------------------------
        # Print information
        # ----------------------------------------------------

        if len(range_axis_m) > 1:

            range_resolution = (
                range_axis_m[1]
                - range_axis_m[0]
            )

        else:

            range_resolution = 0.0

        print(
            f"\n{chip_name} range information:"
        )

        print(
            f"    Range resolution : "
            f"{range_resolution:.6f} m"
        )

        print(
            f"    Maximum range    : "
            f"{range_axis_m[-1]:.6f} m"
        )

        print(
            f"    Range FFT shape  : "
            f"{range_fft.shape}"
        )


    # ========================================================
    # PLOT EACH CHIP
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PLOTTING INDIVIDUAL CHIPS")
    print("=" * 70)

    for chip_name, combined_db in (
        all_range_db.items()
    ):

        plt.figure(
            figsize=(12, 5)
        )

        plt.plot(
            range_axis_m,
            combined_db,
            linewidth=1.5
        )

        plt.xlabel(
            "Range (m)"
        )

        plt.ylabel(
            "Normalized Power (dB)"
        )

        plt.title(
            f"Range FFT - {chip_name}"
        )

        plt.grid(
            True,
            alpha=0.3
        )

        plt.xlim(
            0,
            range_axis_m[-1]
        )

        plt.ylim(
            PLOT_DB_FLOOR,
            5
        )

        plt.tight_layout()

        plt.show()


    # ========================================================
    # COMBINE MASTER + SLAVES
    # ========================================================

    print("\n")
    print("=" * 70)
    print("COMBINING MASTER + SLAVES")
    print("=" * 70)

    total_power = None

    for chip_name, chip_power in (
        all_range_power.items()
    ):

        print(
            f"Adding {chip_name}"
        )

        if total_power is None:

            total_power = (
                chip_power.copy()
            )

        else:

            total_power += chip_power

    # --------------------------------------------------------
    # Convert final combined power to dB
    # --------------------------------------------------------

    total_db = power_to_db(
        total_power
    )

    # ========================================================
    # PLOT FINAL CASCADE RANGE FFT
    # ========================================================

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        range_axis_m,
        total_db,
        linewidth=1.8
    )

    plt.xlabel(
        "Range (m)"
    )

    plt.ylabel(
        "Normalized Power (dB)"
    )

    plt.title(
        "Combined Range FFT - "
        "Master + Slaves"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.xlim(
        0,
        range_axis_m[-1]
    )

    plt.ylim(
        PLOT_DB_FLOOR,
        5
    )

    plt.tight_layout()

    plt.show()


    # ========================================================
    # FIND IMPORTANT RANGE PEAKS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("IMPORTANT RANGE DETECTIONS")
    print("=" * 70)

    peaks, properties = find_peaks(
        total_db,
        height=PEAK_THRESHOLD_DB,
        distance=MIN_PEAK_DISTANCE
    )

    peak_ranges = (
        range_axis_m[peaks]
    )

    peak_powers = (
        total_db[peaks]
    )

    # --------------------------------------------------------
    # Sort strongest first
    # --------------------------------------------------------

    order = np.argsort(
        peak_powers
    )[::-1]

    peak_ranges = (
        peak_ranges[order]
    )

    peak_powers = (
        peak_powers[order]
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    if len(peak_ranges) == 0:

        print(
            "No peaks found above "
            f"{PEAK_THRESHOLD_DB} dB."
        )

    else:

        for i, (
            rng,
            power
        ) in enumerate(
            zip(
                peak_ranges,
                peak_powers
            ),
            start=1
        ):

            print(
                f"{i:2d}. "
                f"Range = {rng:.3f} m    "
                f"Power = {power:.2f} dB"
            )


    # ========================================================
    # PLOT DETECTED PEAKS
    # ========================================================

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        range_axis_m,
        total_db,
        linewidth=1.5,
        label="Combined spectrum"
    )

    if len(peak_ranges) > 0:

        plt.scatter(
            peak_ranges,
            peak_powers,
            s=50,
            marker="x",
            label="Detected peaks"
        )

        # ----------------------------------------------------
        # Add range labels
        # ----------------------------------------------------

        for rng, power in zip(
            peak_ranges,
            peak_powers
        ):

            plt.annotate(
                f"{rng:.2f} m",
                (rng, power),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8
            )

    plt.xlabel(
        "Range (m)"
    )

    plt.ylabel(
        "Normalized Power (dB)"
    )

    plt.title(
        "Combined Cascade Range FFT "
        "with Detected Ranges"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.xlim(
        0,
        range_axis_m[-1]
    )

    plt.ylim(
        PLOT_DB_FLOOR,
        5
    )

    plt.legend()

    plt.tight_layout()

    plt.show()


    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"Processed chips: "
        f"{list(FILES.keys())}"
    )

    print(
        f"Range resolution: "
        f"{range_axis_m[1] - range_axis_m[0]:.4f} m"
    )

    print(
        f"Maximum displayed range: "
        f"{range_axis_m[-1]:.3f} m"
    )

    print(
        f"Number of important ranges: "
        f"{len(peak_ranges)}"
    )

    if len(peak_ranges) > 0:

        print(
            "\nDetected ranges:"
        )

        for rng in peak_ranges:

            print(
                f"    {rng:.3f} m"
            )

    
# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()


