import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

FILES = {
    "master": r"y_door_in\master_0000_data.bin",   # AWR1
    "slave1": r"y_door_in\slave1_0000_data.bin",   # AWR2
    "slave2": r"y_door_in\slave2_0000_data.bin",   # AWR3
    "slave3": r"y_door_in\slave3_0000_data.bin",   # AWR4
}

# FILES = {
#     "master": r"inbox_small_range\master_0000_data.bin",
#     "slave1": r"inbox_small_range\slave1_0000_data.bin",
#     "slave2": r"inbox_small_range\slave2_0000_data.bin",
#     "slave3": r"inbox_small_range\slave3_0000_data.bin",
# }



# ============================================================
# RADAR PROFILE
# ============================================================

START_FREQ_HZ = 77e9

# 79.0327 MHz/us
SLOPE_HZ_PER_S = 79.0327e12

FS_HZ = 8e6

NUM_ADC_SAMPLES = 256
NUM_RX = 4


# ============================================================
# MIMO FRAME
# ============================================================

NUM_TX = 12
NUM_LOOPS = 16

CHIRPS_PER_FRAME = NUM_TX * NUM_LOOPS


# ============================================================
# FFT
# ============================================================

NFFT_RANGE = 256
NFFT_DOPPLER = NUM_LOOPS


# ============================================================
# RANGE
# ============================================================

RANGE_MIN_M = 0.0
RANGE_MAX_M = 15.2


# ============================================================
# ANGLE
# ============================================================

ANGLE_MIN_DEG = -90.0
ANGLE_MAX_DEG = 90.0

NUM_ANGLE_BINS = 361


# ============================================================
# PHYSICAL CONSTANTS
# ============================================================

C0 = 299792458.0

LAMBDA = C0 / START_FREQ_HZ


# ============================================================
# WINDOWS
# ============================================================

USE_RANGE_WINDOW = True
USE_DOPPLER_WINDOW = True


# ============================================================
# DISPLAY
# ============================================================

DB_FLOOR = -45.0


# ============================================================
# IMPORTANT:
#
# TI MMWCAS-RF-EVM AZIMUTH TX POSITIONS
#
# The 9 azimuth TX positions from the antenna-offset layout
# are:
#
#   0, 4, 8, 12, 16, 20, 24, 28, 32
#
# Units are half-wavelength grid positions.
#
# The 3 remaining master TXs are elevation TXs:
#
#   9, 10, 11
#
# based on the assumed global chirp ordering below.
#
# ============================================================

TX_POSITION_BY_CHIRP = np.array([
    9,    # AWR1 TX0 - elevation
    10,   # AWR1 TX1 - elevation
    11,   # AWR1 TX2 - elevation

    24,   # AWR2 TX0 - azimuth
    28,   # AWR2 TX1 - azimuth
    32,   # AWR2 TX2 - azimuth

    12,   # AWR3 TX0 - azimuth
    16,   # AWR3 TX1 - azimuth
    20,   # AWR3 TX2 - azimuth

    0,    # AWR4 TX0 - azimuth
    4,    # AWR4 TX1 - azimuth
    8     # AWR4 TX2 - azimuth
], dtype=int)


# ------------------------------------------------------------
# Azimuth chirps = all except the first three elevation TXs
# ------------------------------------------------------------

AZIMUTH_TX_IDS = np.array([
    3, 4, 5, 6, 7, 8, 9, 10, 11
], dtype=int)


# ============================================================
# RX POSITIONS
#
# From TI antenna offset layout.
#
# Values are in half-wavelength grid units.
#
# Physical device order:
#
# AWR1 = master
# AWR2 = slave1
# AWR3 = slave2
# AWR4 = slave3
#
# ============================================================

RX_POSITION_BY_CHIP = {

    # AWR1
    "master": np.array([
        11, 12, 13, 14
    ], dtype=int),

    # AWR2
    "slave1": np.array([
        50, 51, 52, 53
    ], dtype=int),

    # AWR3
    "slave2": np.array([
        46, 47, 48, 49
    ], dtype=int),

    # AWR4
    "slave3": np.array([
        0, 1, 2, 3
    ], dtype=int),
}


# ============================================================
# VIRTUAL AZIMUTH ARRAY
#
# 9 TX + 16 RX creates 86 unique azimuth positions:
#
# 0 ... 85
#
# Each grid step = lambda/2.
# ============================================================

TX_AZ_POSITIONS = TX_POSITION_BY_CHIRP[
    AZIMUTH_TX_IDS
]

RX_ALL_POSITIONS = np.concatenate([
    RX_POSITION_BY_CHIP[name]
    for name in FILES.keys()
])

VIRTUAL_POSITIONS = np.arange(
    86,
    dtype=int
)


# ============================================================
# PRINT ARRAY INFORMATION
# ============================================================

def print_array_configuration():

    print("\n")
    print("=" * 75)
    print("AZIMUTH VIRTUAL ARRAY CONFIGURATION")
    print("=" * 75)

    print(
        "Azimuth TX positions "
        "(lambda/2 grid):"
    )

    print(
        TX_AZ_POSITIONS
    )

    print(
        "\nRX positions by chip:"
    )

    for chip_name in FILES.keys():

        print(
            f"{chip_name:8s}: "
            f"{RX_POSITION_BY_CHIP[chip_name]}"
        )

    unique_positions = sorted(
        set(
            int(tx_pos + rx_pos)
            for tx_pos in TX_AZ_POSITIONS
            for chip_name in FILES.keys()
            for rx_pos in RX_POSITION_BY_CHIP[
                chip_name
            ]
        )
    )

    print(
        "\nUnique virtual positions:"
    )

    print(
        unique_positions
    )

    print(
        "\nNumber of unique virtual elements:",
        len(unique_positions)
    )

    if len(unique_positions) != 86:

        raise ValueError(
            "Virtual array is not 86 elements. "
            "Check TX/RX antenna positions."
        )

    print(
        "\nVirtual array:"
    )

    print(
        "0, 1, 2, ..., 85"
    )

    print(
        "Spacing = lambda/2"
    )

    print(
        f"lambda = {LAMBDA * 1e3:.4f} mm"
    )


# ============================================================
# RANGE AXIS
# ============================================================

def make_range_axis():

    freq_axis = (
        np.arange(NFFT_RANGE)
        * FS_HZ
        / NFFT_RANGE
    )

    range_axis = (
        C0
        * freq_axis
        / (
            2.0
            * SLOPE_HZ_PER_S
        )
    )

    return (
        freq_axis,
        range_axis
    )


# ============================================================
# DOPPLER AXIS
# ============================================================

def make_doppler_axis():

    chirp_time_s = (
        (40.0 + 5.0)
        * 1e-6
    )

    same_tx_period_s = (
        NUM_TX
        * chirp_time_s
    )

    slow_time_prf = (
        1.0
        / same_tx_period_s
    )

    doppler_freq = np.fft.fftshift(
        np.fft.fftfreq(
            NFFT_DOPPLER,
            d=1.0 / slow_time_prf
        )
    )

    velocity_axis = (
        doppler_freq
        * LAMBDA
        / 2.0
    )

    return (
        doppler_freq,
        velocity_axis
    )


# ============================================================
# LOAD BINARY DATA
# ============================================================

def load_dca1000_iq(
    file_path
):

    if not os.path.exists(
        file_path
    ):

        raise FileNotFoundError(
            f"File not found:\n{file_path}"
        )

    raw = np.fromfile(
        file_path,
        dtype=np.int16
    )

    print(
        f"\nLoading: {file_path}"
    )

    print(
        f"Raw int16 values: {raw.size}"
    )

    # --------------------------------------------------------
    # I/Q reconstruction
    #
    # I0 Q0 I1 Q1 ...
    # --------------------------------------------------------

    if raw.size % 2 != 0:

        raw = raw[:-1]

        print(
            "WARNING: dropped final int16 value"
        )

    I = raw[0::2].astype(
        np.float32
    )

    Q = raw[1::2].astype(
        np.float32
    )

    complex_data = (
        I + 1j * Q
    )

    # --------------------------------------------------------
    # Samples per chirp
    # --------------------------------------------------------

    samples_per_chirp = (
        NUM_ADC_SAMPLES
        * NUM_RX
    )

    if (
        complex_data.size
        % samples_per_chirp
        != 0
    ):

        raise ValueError(
            f"Cannot reshape {file_path}\n"
            f"Complex samples = "
            f"{complex_data.size}\n"
            f"Samples/chirp = "
            f"{samples_per_chirp}\n"
            f"Remainder = "
            f"{complex_data.size % samples_per_chirp}"
        )

    num_chirps = (
        complex_data.size
        // samples_per_chirp
    )

    print(
        f"Complex samples: "
        f"{complex_data.size}"
    )

    print(
        f"Number of chirps: "
        f"{num_chirps}"
    )

    # --------------------------------------------------------
    # RX-interleaved layout
    #
    # Raw:
    #
    # sample0: RX0 RX1 RX2 RX3
    # sample1: RX0 RX1 RX2 RX3
    # ...
    #
    # First:
    #
    # (chirp, sample, RX)
    #
    # Then:
    #
    # (chirp, RX, sample)
    # --------------------------------------------------------

    adc_cube = complex_data.reshape(
        num_chirps,
        NUM_ADC_SAMPLES,
        NUM_RX
    ).transpose(
        0,
        2,
        1
    )

    print(
        f"ADC cube shape: "
        f"{adc_cube.shape}"
    )

    print(
        "Format = "
        "(chirp, RX, ADC sample)"
    )

    return adc_cube


# ============================================================
# RANGE FFT
# ============================================================

def compute_range_fft(
    adc_frame
):

    if USE_RANGE_WINDOW:

        window = np.hanning(
            NUM_ADC_SAMPLES
        )

        adc_frame = (
            adc_frame
            * window[
                None,
                None,
                :
            ]
        )

    range_fft = np.fft.fft(
        adc_frame,
        n=NFFT_RANGE,
        axis=-1
    )

    return range_fft


# ============================================================
# ZERO-DOPPLER COMPLEX DATA
# ============================================================

def get_zero_doppler_data(
    range_fft_frame
):
    """
    Input:
        range_fft_frame:
            (192 chirps, 4 RX, range)

    Output:
        zero_doppler:
            (12 TX, 4 RX, range)

    This preserves COMPLEX phase information, which is
    necessary for angle estimation.
    """

    if (
        range_fft_frame.shape[0]
        != CHIRPS_PER_FRAME
    ):

        raise ValueError(
            "Unexpected chirp count in frame."
        )

    # --------------------------------------------------------
    # Reshape:
    #
    # (loop, TX, RX, range)
    # --------------------------------------------------------

    temp = range_fft_frame.reshape(
        NUM_LOOPS,
        NUM_TX,
        NUM_RX,
        NFFT_RANGE
    )

    # --------------------------------------------------------
    # Move TX before loop:
    #
    # (TX, loop, RX, range)
    # --------------------------------------------------------

    temp = temp.transpose(
        1,
        0,
        2,
        3
    )

    # --------------------------------------------------------
    # Doppler window
    # --------------------------------------------------------

    if USE_DOPPLER_WINDOW:

        doppler_window = np.hanning(
            NUM_LOOPS
        )

        temp = (
            temp
            * doppler_window[
                None,
                :,
                None,
                None
            ]
        )

    # --------------------------------------------------------
    # Doppler FFT along loop dimension
    # --------------------------------------------------------

    doppler_fft = np.fft.fft(
        temp,
        n=NFFT_DOPPLER,
        axis=1
    )

    doppler_fft = np.fft.fftshift(
        doppler_fft,
        axes=1
    )

    # --------------------------------------------------------
    # Center bin = zero Doppler
    # --------------------------------------------------------

    zero_index = (
        NFFT_DOPPLER // 2
    )

    zero_doppler = (
        doppler_fft[
            :,
            zero_index,
            :,
            :
        ]
    )

    # Shape:
    #
    # (12 TX, 4 RX, range)

    return zero_doppler


# ============================================================
# BUILD VIRTUAL ARRAY
# ============================================================

def build_virtual_array(
    zero_doppler_by_chip
):
    """
    Build 86-element virtual azimuth array.

    Input:
        dictionary:
            chip -> (12 TX, 4 RX, range)

    Output:
        virtual_data:
            (86 virtual elements, range)

    Duplicate virtual positions are coherently averaged.
    """

    num_range_bins = NFFT_RANGE

    virtual_data = np.zeros(
        (
            86,
            num_range_bins
        ),
        dtype=np.complex128
    )

    virtual_count = np.zeros(
        86,
        dtype=np.int32
    )

    # --------------------------------------------------------
    # Process only the 9 AZIMUTH TXs
    # --------------------------------------------------------

    for tx_id in AZIMUTH_TX_IDS:

        tx_pos = (
            TX_POSITION_BY_CHIRP[
                tx_id
            ]
        )

        # ----------------------------------------------------
        # All four AWR devices receive every TX chirp
        # ----------------------------------------------------

        for chip_name in FILES.keys():

            rx_positions = (
                RX_POSITION_BY_CHIP[
                    chip_name
                ]
            )

            chip_data = (
                zero_doppler_by_chip[
                    chip_name
                ]
            )

            for rx_id in range(
                NUM_RX
            ):

                rx_pos = (
                    rx_positions[
                        rx_id
                    ]
                )

                virtual_pos = (
                    tx_pos
                    + rx_pos
                )

                if (
                    virtual_pos < 0
                    or virtual_pos >= 86
                ):

                    raise ValueError(
                        f"Virtual position "
                        f"{virtual_pos} outside 0...85"
                    )

                signal = (
                    chip_data[
                        tx_id,
                        rx_id,
                        :
                    ]
                )

                virtual_data[
                    virtual_pos,
                    :
                ] += signal

                virtual_count[
                    virtual_pos
                ] += 1

    # --------------------------------------------------------
    # Coherent average of overlapping virtual channels
    # --------------------------------------------------------

    for pos in range(86):

        if virtual_count[pos] > 0:

            virtual_data[pos, :] /= (
                virtual_count[pos]
            )

    return virtual_data


# ============================================================
# RANGE SELECTION
# ============================================================

def select_range(
    range_axis,
    virtual_data
):

    mask = (
        (range_axis >= RANGE_MIN_M)
        &
        (range_axis <= RANGE_MAX_M)
    )

    return (
        range_axis[mask],
        virtual_data[:, mask]
    )


# ============================================================
# RANGE-ANGLE BEAMFORMING
# ============================================================

def compute_range_angle_map(
    virtual_data
):
    """
    Conventional Bartlett / delay-and-sum beamforming.

    Virtual array:
        positions = 0 ... 85
        spacing = lambda/2

    Steering phase:
        exp(-j*pi*m*sin(theta))

    Output:
        angle x range
    """

    num_virtual = 86

    # --------------------------------------------------------
    # Virtual element coordinate in wavelengths
    #
    # Every array step = lambda/2
    # --------------------------------------------------------

    virtual_pos_lambda = (
        np.arange(
            num_virtual
        )
        * 0.5
    )

    # --------------------------------------------------------
    # Angle grid
    # --------------------------------------------------------

    angle_axis_deg = np.linspace(
        ANGLE_MIN_DEG,
        ANGLE_MAX_DEG,
        NUM_ANGLE_BINS
    )

    angle_axis_rad = np.deg2rad(
        angle_axis_deg
    )

    sin_theta = np.sin(
        angle_axis_rad
    )

    # --------------------------------------------------------
    # Steering matrix
    #
    # Shape:
    #     virtual element × angle
    # --------------------------------------------------------

    steering = np.exp(
        -1j
        * 2.0
        * np.pi
        * virtual_pos_lambda[
            :, None
        ]
        * sin_theta[
            None, :
        ]
    )

    # --------------------------------------------------------
    # Beamforming
    #
    # virtual_data:
    #     virtual × range
    #
    # steering.conj().T:
    #     angle × virtual
    #
    # output:
    #     angle × range
    # --------------------------------------------------------

    beamformed = (
        steering.conj().T
        @ virtual_data
    )

    # Normalize by number of virtual elements

    beamformed /= (
        np.sqrt(num_virtual)
    )

    # --------------------------------------------------------
    # Power
    # --------------------------------------------------------

    ra_power = (
        np.abs(beamformed)
        ** 2
    )

    return (
        angle_axis_deg,
        ra_power
    )


# ============================================================
# NORMALIZE DB
# ============================================================

def power_to_db(
    power
):

    db = (
        10.0
        * np.log10(
            power + 1e-18
        )
    )

    db -= np.max(
        db
    )

    db = np.maximum(
        db,
        DB_FLOOR
    )

    return db


# ============================================================
# PLOT VIRTUAL ARRAY
# ============================================================

def plot_virtual_array():

    plt.figure(
        figsize=(14, 2.5)
    )

    plt.scatter(
        VIRTUAL_POSITIONS * 0.5,
        np.zeros(86),
        s=25
    )

    plt.xlabel(
        "Virtual element position (lambda)"
    )

    plt.yticks([])

    plt.title(
        "MMWCAS-RF-EVM 86-Element "
        "Azimuth Virtual Array"
    )

    plt.grid(
        True,
        axis="x",
        alpha=0.3
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        + "=" * 75
    )

    print(
        "MMWCAS-RF-EVM RANGE-ANGLE PROCESSING"
    )

    print(
        "=" * 75
    )

    # --------------------------------------------------------
    # Array configuration
    # --------------------------------------------------------

    print_array_configuration()

    # --------------------------------------------------------
    # Range axis
    # --------------------------------------------------------

    freq_axis, range_axis = (
        make_range_axis()
    )

    print(
        "\nRange bin spacing:"
    )

    print(
        f"{range_axis[1] - range_axis[0]:.6f} m"
    )

    print(
        f"Maximum range axis: "
        f"{range_axis[-1]:.3f} m"
    )

    # --------------------------------------------------------
    # Doppler axis
    # --------------------------------------------------------

    doppler_freq, velocity_axis = (
        make_doppler_axis()
    )

    print(
        "\nZero-Doppler processing:"
    )

    print(
        f"Doppler bins: "
        f"{NFFT_DOPPLER}"
    )

    print(
        f"Velocity bin spacing: "
        f"{(velocity_axis[1] - velocity_axis[0]):.4f} m/s"
    )

    # --------------------------------------------------------
    # Optional:
    # visualize the 86-element array
    # --------------------------------------------------------

    plot_virtual_array()

    # --------------------------------------------------------
    # Load all four chips
    # --------------------------------------------------------

    adc_data = {}

    for chip_name, file_path in FILES.items():

        adc_data[
            chip_name
        ] = load_dca1000_iq(
            file_path
        )

    # --------------------------------------------------------
    # Check frame counts
    # --------------------------------------------------------

    num_frames_list = []

    for chip_name in FILES.keys():

        n_chirps = (
            adc_data[
                chip_name
            ].shape[0]
        )

        if (
            n_chirps
            % CHIRPS_PER_FRAME
            != 0
        ):

            raise ValueError(
                f"{chip_name}: "
                f"{n_chirps} chirps is not "
                f"divisible by {CHIRPS_PER_FRAME}."
            )

        num_frames = (
            n_chirps
            // CHIRPS_PER_FRAME
        )

        num_frames_list.append(
            num_frames
        )

    if len(
        set(num_frames_list)
    ) != 1:

        raise ValueError(
            "Different numbers of frames "
            "were found between cascade files."
        )

    num_frames = num_frames_list[0]

    print(
        "\nNumber of complete frames:"
    )

    print(
        num_frames
    )

    # --------------------------------------------------------
    # Average Range-Angle POWER across frames
    # --------------------------------------------------------

    final_ra_power = None

    # --------------------------------------------------------
    # Process each frame
    # --------------------------------------------------------

    for frame_id in range(
        num_frames
    ):

        print(
            "\n"
            + "=" * 75
        )

        print(
            f"PROCESSING FRAME "
            f"{frame_id + 1}/{num_frames}"
        )

        print(
            "=" * 75
        )

        # ----------------------------------------------------
        # Zero-Doppler data from each chip
        # ----------------------------------------------------

        zero_doppler_by_chip = {}

        for chip_name in FILES.keys():

            start = (
                frame_id
                * CHIRPS_PER_FRAME
            )

            end = (
                start
                + CHIRPS_PER_FRAME
            )

            adc_frame = (
                adc_data[
                    chip_name
                ][
                    start:end
                ]
            )

            print(
                f"{chip_name}: "
                f"frame shape = "
                f"{adc_frame.shape}"
            )

            # ------------------------------------------------
            # Range FFT
            # ------------------------------------------------

            range_fft = (
                compute_range_fft(
                    adc_frame
                )
            )

            # ------------------------------------------------
            # Zero-Doppler complex data
            # ------------------------------------------------

            zero_doppler = (
                get_zero_doppler_data(
                    range_fft
                )
            )

            zero_doppler_by_chip[
                chip_name
            ] = zero_doppler

        # ----------------------------------------------------
        # Build 86-element virtual array
        # ----------------------------------------------------

        virtual_data = (
            build_virtual_array(
                zero_doppler_by_chip
            )
        )

        # ----------------------------------------------------
        # Select desired range
        # ----------------------------------------------------

        selected_range_axis, selected_virtual = (
            select_range(
                range_axis,
                virtual_data
            )
        )

        # ----------------------------------------------------
        # Compute range-angle map
        # ----------------------------------------------------

        angle_axis_deg, ra_power = (
            compute_range_angle_map(
                selected_virtual
            )
        )

        # ----------------------------------------------------
        # Accumulate power
        # ----------------------------------------------------

        if final_ra_power is None:

            final_ra_power = (
                ra_power.copy()
            )

        else:

            final_ra_power += (
                ra_power
            )

    # ========================================================
    # FINAL RANGE-ANGLE MAP
    # ========================================================

    final_ra_db = (
        power_to_db(
            final_ra_power
        )
    )

    # ========================================================
    # PLOT
    # ========================================================

    plt.figure(
        figsize=(13, 7)
    )

    plt.imshow(
        final_ra_db,
        aspect="auto",
        origin="lower",
        extent=[
            selected_range_axis[0],
            selected_range_axis[-1],
            angle_axis_deg[0],
            angle_axis_deg[-1]
        ],
        vmin=DB_FLOOR,
        vmax=0,
        cmap="turbo"
    )

    plt.colorbar(
        label="Normalized Power (dB)"
    )

    plt.xlabel(
        "Range (m)"
    )

    plt.ylabel(
        "Azimuth Angle (degrees)"
    )

    plt.title(
        "MMWCAS-RF-EVM Combined "
        "Range-Angle Map"
    )

    plt.xlim(
        RANGE_MIN_M,
        RANGE_MAX_M
    )

    plt.ylim(
        ANGLE_MIN_DEG,
        ANGLE_MAX_DEG
    )

    plt.tight_layout()

    plt.show()

    # ========================================================
    # SAVE
    # ========================================================

    plt.figure(
        figsize=(13, 7)
    )

    plt.imshow(
        final_ra_db,
        aspect="auto",
        origin="lower",
        extent=[
            selected_range_axis[0],
            selected_range_axis[-1],
            angle_axis_deg[0],
            angle_axis_deg[-1]
        ],
        vmin=DB_FLOOR,
        vmax=0,
        cmap="turbo"
    )

    plt.colorbar(
        label="Normalized Power (dB)"
    )

    plt.xlabel(
        "Range (m)"
    )

    plt.ylabel(
        "Azimuth Angle (degrees)"
    )

    plt.title(
        "MMWCAS-RF-EVM Combined "
        "Range-Angle Map"
    )

    plt.xlim(
        RANGE_MIN_M,
        RANGE_MAX_M
    )

    plt.ylim(
        ANGLE_MIN_DEG,
        ANGLE_MAX_DEG
    )

    plt.tight_layout()

    plt.savefig(
        "range_angle_map_cascade.png",
        dpi=300
    )

    plt.close()

    # ========================================================
    # FIND STRONGEST RANGE-ANGLE POINTS
    # ========================================================

    max_index = np.unravel_index(
        np.argmax(
            final_ra_power
        ),
        final_ra_power.shape
    )

    max_angle_index = (
        max_index[0]
    )

    max_range_index = (
        max_index[1]
    )

    strongest_angle = (
        angle_axis_deg[
            max_angle_index
        ]
    )

    strongest_range = (
        selected_range_axis[
            max_range_index
        ]
    )

    strongest_power = (
        final_ra_db[
            max_angle_index,
            max_range_index
        ]
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "STRONGEST RANGE-ANGLE RESPONSE"
    )

    print(
        "=" * 75
    )

    print(
        f"Range  : "
        f"{strongest_range:.3f} m"
    )

    print(
        f"Angle  : "
        f"{strongest_angle:.3f} deg"
    )

    print(
        f"Power  : "
        f"{strongest_power:.3f} dB"
    )

    print(
        "\nSaved:"
    )

    print(
        "range_angle_map_cascade.png"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()