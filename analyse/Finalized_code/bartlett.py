import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

FILES = {
    "master": r"b_door_in\master_0000_data.bin",
    "slave1": r"b_door_in\slave1_0000_data.bin",
    "slave2": r"b_door_in\slave2_0000_data.bin",
    "slave3": r"b_door_in\slave3_0000_data.bin",
}


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

CHIRPS_PER_FRAME = (
    NUM_TX * NUM_LOOPS
)


# ============================================================
# FFT
# ============================================================

NFFT_RANGE = 256


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
# 2-D CA-CFAR
# ============================================================

TRAINING_CELLS = (
    4,   # angle
    8    # range
)

GUARD_CELLS = (
    2,   # angle
    2    # range
)

PFA = 1e-4

MIN_DETECTION_DB = -35.0

MIN_DETECTION_RANGE_M = 0.10
MIN_DETECTION_ANGLE_DEG = 1.0

MAX_DETECTIONS = 100


# ============================================================
# DISPLAY
# ============================================================

DB_FLOOR = -45.0


# ============================================================
# PHYSICAL CONSTANTS
# ============================================================

C0 = 299792458.0

LAMBDA = (
    C0 / START_FREQ_HZ
)

HALF_WAVELENGTH = (
    LAMBDA / 2.0
)


# ============================================================
# WINDOW
# ============================================================

USE_RANGE_WINDOW = True


# ============================================================
# TX POSITIONS
#
# Units = lambda/2 grid positions
# ============================================================

TX_POSITION_BY_CHIRP = np.array([
    9,
    10,
    11,

    24,
    28,
    32,

    12,
    16,
    20,

    0,
    4,
    8
], dtype=float)


# ============================================================
# AZIMUTH TX CHANNELS
# ============================================================

AZIMUTH_TX_IDS = np.array([
    3, 4, 5,
    6, 7, 8,
    9, 10, 11
], dtype=int)


# ============================================================
# RX POSITIONS
#
# Units = lambda/2 grid positions
# ============================================================

RX_POSITION_BY_CHIP = {

    "master": np.array([
        11,
        12,
        13,
        14
    ], dtype=float),

    "slave1": np.array([
        50,
        51,
        52,
        53
    ], dtype=float),

    "slave2": np.array([
        46,
        47,
        48,
        49
    ], dtype=float),

    "slave3": np.array([
        0,
        1,
        2,
        3
    ], dtype=float),
}


# ============================================================
# PHYSICAL TX/RX POSITIONS
# ============================================================

TX_POSITION_M = (
    TX_POSITION_BY_CHIRP
    * HALF_WAVELENGTH
)

RX_POSITION_M = {}

for chip_name in FILES.keys():

    RX_POSITION_M[chip_name] = (
        RX_POSITION_BY_CHIP[chip_name]
        * HALF_WAVELENGTH
    )


# ============================================================
# TX-RX SPATIAL CHANNELS
#
# Each channel is one TX-RX pair.
#
# We preserve the actual physical TX+RX separation.
# ============================================================

VIRTUAL_PAIRS = []

for tx_id in AZIMUTH_TX_IDS:

    tx_x = TX_POSITION_M[tx_id]

    for chip_name in FILES.keys():

        rx_positions = (
            RX_POSITION_M[chip_name]
        )

        for rx_id in range(NUM_RX):

            rx_x = rx_positions[rx_id]

            pair_x = tx_x + rx_x

            VIRTUAL_PAIRS.append({

                "tx_id": int(tx_id),

                "chip": chip_name,

                "rx_id": int(rx_id),

                "tx_x_m": float(tx_x),

                "rx_x_m": float(rx_x),

                "pair_x_m": float(pair_x)
            })


NUM_VIRTUAL_PAIRS = len(
    VIRTUAL_PAIRS
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
# LOAD RAW ADC
# ============================================================

def load_dca1000_iq(file_path):

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"File not found:\n{file_path}"
        )

    raw = np.fromfile(
        file_path,
        dtype=np.int16
    )

    print(f"\nLoading: {file_path}")
    print(
        f"Raw int16 values: {raw.size}"
    )

    if raw.size % 2 != 0:

        raw = raw[:-1]

        print(
            "WARNING: odd number of int16 "
            "samples; last value removed."
        )

    # --------------------------------------------------------
    # I/Q reconstruction
    #
    # I0 Q0 I1 Q1 ...
    # --------------------------------------------------------

    I = raw[0::2].astype(
        np.float32
    )

    Q = raw[1::2].astype(
        np.float32
    )

    complex_data = I + 1j * Q

    # --------------------------------------------------------
    # RX-interleaved capture
    #
    # sample0: RX0 RX1 RX2 RX3
    # sample1: RX0 RX1 RX2 RX3
    # ...
    #
    # -> (chirp, RX, ADC sample)
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
            f"Cannot reshape file:\n"
            f"{file_path}\n"
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

    adc_cube = (
        complex_data
        .reshape(
            num_chirps,
            NUM_ADC_SAMPLES,
            NUM_RX
        )
        .transpose(
            0,
            2,
            1
        )
    )

    print(
        f"Number of chirps: "
        f"{num_chirps}"
    )

    print(
        f"ADC cube shape: "
        f"{adc_cube.shape}"
    )

    return adc_cube


# ============================================================
# RANGE FFT
# ============================================================

def compute_range_fft(adc_frame):

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
# ARRANGE MIMO DATA
# ============================================================

def arrange_mimo_data(range_fft):

    if (
        range_fft.shape[0]
        != CHIRPS_PER_FRAME
    ):

        raise ValueError(
            f"Expected "
            f"{CHIRPS_PER_FRAME} chirps, "
            f"got {range_fft.shape[0]}"
        )

    # --------------------------------------------------------
    # Assumed chirp ordering:
    #
    # TX0 TX1 ... TX11
    # TX0 TX1 ... TX11
    # ...
    #
    # Output:
    #
    # loop, TX, RX, range
    # --------------------------------------------------------

    return range_fft.reshape(
        NUM_LOOPS,
        NUM_TX,
        NUM_RX,
        NFFT_RANGE
    )


# ============================================================
# BUILD SPATIAL SNAPSHOTS
# ============================================================

def build_spatial_snapshots(
    mimo_data_by_chip
):
    """
    Output:

        snapshots

        shape =
        (loops,
         TX-RX spatial channels,
         range)

    Each loop is one spatial snapshot.
    """

    snapshots = np.zeros(
        (
            NUM_LOOPS,
            NUM_VIRTUAL_PAIRS,
            NFFT_RANGE
        ),
        dtype=np.complex128
    )

    for pair_id, pair in enumerate(
        VIRTUAL_PAIRS
    ):

        tx_id = pair["tx_id"]
        chip_name = pair["chip"]
        rx_id = pair["rx_id"]

        snapshots[
            :,
            pair_id,
            :
        ] = (
            mimo_data_by_chip[
                chip_name
            ][
                :,
                tx_id,
                rx_id,
                :
            ]
        )

    return snapshots


# ============================================================
# BARTLETT RANGE-ANGLE MAP
# ============================================================

def compute_bartlett_map(
    spatial_snapshots,
    pair_x_positions_m
):
    """
    Conventional Bartlett / delay-and-sum beamforming.

    Spatial snapshot:
        x

    Steering vector:
        a(theta)

    Bartlett response:

        P(theta)
        =
        |a^H x|^2 / N

    Here the steering vector explicitly uses:

        x_pair = x_TX + x_RX

    so the actual TX-RX spatial separation is included.
    """

    num_snapshots = (
        spatial_snapshots.shape[0]
    )

    num_channels = (
        spatial_snapshots.shape[1]
    )

    num_ranges = (
        spatial_snapshots.shape[2]
    )

    # --------------------------------------------------------
    # Angle axis
    # --------------------------------------------------------

    angle_axis_deg = np.linspace(
        ANGLE_MIN_DEG,
        ANGLE_MAX_DEG,
        NUM_ANGLE_BINS
    )

    angle_axis_rad = np.deg2rad(
        angle_axis_deg
    )

    # --------------------------------------------------------
    # Wavenumber
    # --------------------------------------------------------

    k = (
        2.0
        * np.pi
        / LAMBDA
    )

    # --------------------------------------------------------
    # Physical bistatic steering
    #
    # a_m(theta)
    # =
    # exp(
    #   -j*k*x_pair_m*sin(theta)
    # )
    # --------------------------------------------------------

    steering = np.exp(
        -1j
        * k
        * pair_x_positions_m[
            :, None
        ]
        * np.sin(
            angle_axis_rad
        )[None, :]
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    bartlett_power = np.zeros(
        (
            NUM_ANGLE_BINS,
            num_ranges
        ),
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Average spatial power across loops
    #
    # Bartlett can be evaluated for every snapshot
    # and then averaged.
    # --------------------------------------------------------

    for snapshot_id in range(
        num_snapshots
    ):

        X = (
            spatial_snapshots[
                snapshot_id,
                :,
                :
            ]
        )

        # X:
        # channels × range

        # ----------------------------------------------------
        # Steering^H X
        #
        # angles × range
        # ----------------------------------------------------

        beamformed = (
            steering.conj().T
            @ X
        )

        power = (
            np.abs(
                beamformed
            ) ** 2
        )

        power /= (
            num_channels
        )

        bartlett_power += power

    bartlett_power /= (
        num_snapshots
    )

    return (
        angle_axis_deg,
        bartlett_power
    )


# ============================================================
# SELECT RANGE
# ============================================================

def select_range(
    range_axis,
    data
):

    mask = (
        (range_axis >= RANGE_MIN_M)
        &
        (range_axis <= RANGE_MAX_M)
    )

    return (
        range_axis[mask],
        data[..., mask]
    )


# ============================================================
# 2-D CA-CFAR
# ============================================================

def ca_cfar_2d(
    power_map,
    training_cells=(4, 8),
    guard_cells=(2, 2),
    pfa=1e-4
):
    """
    2-D CA-CFAR.

    Input:
        linear power map
        shape = (angle, range)

    training_cells:
        (angle, range)

    guard_cells:
        (angle, range)
    """

    num_angle, num_range = (
        power_map.shape
    )

    t_angle, t_range = (
        training_cells
    )

    g_angle, g_range = (
        guard_cells
    )

    total_angle = (
        2 * (t_angle + g_angle)
        + 1
    )

    total_range = (
        2 * (t_range + g_range)
        + 1
    )

    total_cells = (
        total_angle
        * total_range
    )

    guard_area = (
        (2 * g_angle + 1)
        *
        (2 * g_range + 1)
    )

    num_training = (
        total_cells
        - guard_area
    )

    if num_training <= 0:

        raise ValueError(
            "Invalid CFAR configuration."
        )

    # --------------------------------------------------------
    # CA-CFAR threshold scaling
    # --------------------------------------------------------

    alpha = (
        num_training
        *
        (
            pfa ** (
                -1.0
                / num_training
            )
            - 1.0
        )
    )

    threshold_map = np.zeros_like(
        power_map
    )

    noise_map = np.zeros_like(
        power_map
    )

    detections = np.zeros(
        power_map.shape,
        dtype=bool
    )

    # --------------------------------------------------------
    # CUT processing
    # --------------------------------------------------------

    a_start = (
        t_angle + g_angle
    )

    a_end = (
        num_angle
        - t_angle
        - g_angle
    )

    r_start = (
        t_range + g_range
    )

    r_end = (
        num_range
        - t_range
        - g_range
    )

    for a in range(
        a_start,
        a_end
    ):

        for r in range(
            r_start,
            r_end
        ):

            a0 = (
                a
                - t_angle
                - g_angle
            )

            a1 = (
                a
                + t_angle
                + g_angle
                + 1
            )

            r0 = (
                r
                - t_range
                - g_range
            )

            r1 = (
                r
                + t_range
                + g_range
                + 1
            )

            window = (
                power_map[
                    a0:a1,
                    r0:r1
                ]
            )

            # ------------------------------------------------
            # Training mask
            # ------------------------------------------------

            mask = np.ones(
                window.shape,
                dtype=bool
            )

            guard_a0 = t_angle

            guard_a1 = (
                t_angle
                + 2 * g_angle
                + 1
            )

            guard_r0 = t_range

            guard_r1 = (
                t_range
                + 2 * g_range
                + 1
            )

            mask[
                guard_a0:guard_a1,
                guard_r0:guard_r1
            ] = False

            noise_power = np.mean(
                window[mask]
            )

            threshold = (
                alpha
                * noise_power
            )

            noise_map[
                a,
                r
            ] = noise_power

            threshold_map[
                a,
                r
            ] = threshold

            if (
                power_map[a, r]
                > threshold
            ):

                detections[
                    a,
                    r
                ] = True

    return (
        detections,
        threshold_map,
        noise_map
    )


# ============================================================
# EXTRACT CFAR CANDIDATES
# ============================================================

def extract_cfar_candidates(
    detections,
    power_map,
    range_axis,
    angle_axis,
    minimum_db=-35.0
):

    power_db = (
        10.0
        * np.log10(
            power_map
            + 1e-18
        )
    )

    power_db -= np.max(
        power_db
    )

    valid = (
        detections
        &
        (
            power_db
            >= minimum_db
        )
    )

    indices = np.argwhere(
        valid
    )

    candidates = []

    for a, r in indices:

        candidates.append({

            "angle_index":
                int(a),

            "range_index":
                int(r),

            "angle_deg":
                float(
                    angle_axis[a]
                ),

            "range_m":
                float(
                    range_axis[r]
                ),

            "power_linear":
                float(
                    power_map[a, r]
                ),

            "power_db":
                float(
                    power_db[a, r]
                )
        })

    # strongest first
    candidates.sort(
        key=lambda x:
            x["power_linear"],
        reverse=True
    )

    return candidates


# ============================================================
# NON-MAXIMUM SUPPRESSION
# ============================================================

def suppress_detections(
    candidates,
    max_detections=100,
    min_range_separation=0.10,
    min_angle_separation=1.0
):

    selected = []

    for candidate in candidates:

        keep = True

        for existing in selected:

            range_close = (
                abs(
                    candidate["range_m"]
                    -
                    existing["range_m"]
                )
                <= min_range_separation
            )

            angle_close = (
                abs(
                    candidate["angle_deg"]
                    -
                    existing["angle_deg"]
                )
                <= min_angle_separation
            )

            if (
                range_close
                and angle_close
            ):

                keep = False
                break

        if not keep:
            continue

        selected.append(
            candidate
        )

        if len(
            selected
        ) >= max_detections:

            break

    return selected


# ============================================================
# X-Y CONVERSION
# ============================================================

def add_xy_coordinates(
    detections
):

    if len(
        detections
    ) == 0:

        return detections

    maximum_power = max(
        d["power_db"]
        for d in detections
    )

    for d in detections:

        theta = np.deg2rad(
            d["angle_deg"]
        )

        R = d["range_m"]

        d["x_m"] = (
            R
            * np.cos(theta)
        )

        d["y_m"] = (
            R
            * np.sin(theta)
        )

        d["relative_power_db"] = (
            d["power_db"]
            - maximum_power
        )

    return detections


# ============================================================
# PLOT BARTLETT MAP
# ============================================================

def plot_bartlett_map(
    range_axis,
    angle_axis,
    bartlett_db
):

    plt.figure(
        figsize=(13, 7)
    )

    plt.imshow(
        bartlett_db,
        aspect="auto",
        origin="lower",
        extent=[
            range_axis[0],
            range_axis[-1],
            angle_axis[0],
            angle_axis[-1]
        ],
        cmap="turbo",
        vmin=DB_FLOOR,
        vmax=0
    )

    plt.colorbar(
        label="Normalized Bartlett Power (dB)"
    )

    plt.xlabel(
        "Range (m)"
    )

    plt.ylabel(
        "Azimuth Angle (deg)"
    )

    plt.title(
        "Bartlett Range-Angle Map"
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


# ============================================================
# PLOT CFAR RANGE-ANGLE
# ============================================================

def plot_cfar_range_angle(
    range_axis,
    angle_axis,
    bartlett_db,
    detections
):

    plt.figure(
        figsize=(13, 7)
    )

    plt.imshow(
        bartlett_db,
        aspect="auto",
        origin="lower",
        extent=[
            range_axis[0],
            range_axis[-1],
            angle_axis[0],
            angle_axis[-1]
        ],
        cmap="turbo",
        vmin=DB_FLOOR,
        vmax=0
    )

    if len(
        detections
    ) > 0:

        ranges = [
            d["range_m"]
            for d in detections
        ]

        angles = [
            d["angle_deg"]
            for d in detections
        ]

        plt.scatter(
            ranges,
            angles,
            facecolors="none",
            edgecolors="white",
            s=90,
            linewidths=1.5,
            marker="o",
            label="CA-CFAR detections"
        )

        for i, d in enumerate(
            detections,
            start=1
        ):

            plt.annotate(
                str(i),
                (
                    d["range_m"],
                    d["angle_deg"]
                ),
                xytext=(
                    5,
                    5
                ),
                textcoords="offset points",
                color="white",
                fontsize=8
            )

    plt.colorbar(
        label="Normalized Bartlett Power (dB)"
    )

    plt.xlabel(
        "Range (m)"
    )

    plt.ylabel(
        "Azimuth Angle (deg)"
    )

    plt.title(
        "Bartlett Range-Angle Map + 2-D CA-CFAR"
    )

    plt.xlim(
        RANGE_MIN_M,
        RANGE_MAX_M
    )

    plt.ylim(
        ANGLE_MIN_DEG,
        ANGLE_MAX_DEG
    )

    if len(
        detections
    ) > 0:

        plt.legend()

    plt.tight_layout()

    plt.show()


# ============================================================
# PLOT X-Y DETECTIONS
# ============================================================

def plot_xy_detections(
    detections
):

    plt.figure(
        figsize=(10, 9)
    )

    if len(
        detections
    ) > 0:

        x = [
            d["x_m"]
            for d in detections
        ]

        y = [
            d["y_m"]
            for d in detections
        ]

        power = [
            d["relative_power_db"]
            for d in detections
        ]

        scatter = plt.scatter(
            x,
            y,
            c=power,
            cmap="turbo",
            s=80
        )

        plt.colorbar(
            scatter,
            label="Relative Bartlett Power (dB)"
        )

        for i, d in enumerate(
            detections,
            start=1
        ):

            plt.annotate(
                str(i),
                (
                    d["x_m"],
                    d["y_m"]
                ),
                xytext=(
                    5,
                    5
                ),
                textcoords="offset points"
            )

    # Radar
    plt.scatter(
        [0],
        [0],
        marker="x",
        s=150,
        linewidths=3,
        label="Radar"
    )

    plt.xlabel(
        "X (m)"
    )

    plt.ylabel(
        "Y (m)"
    )

    plt.title(
        "CA-CFAR Detections in X-Y Coordinates"
    )

    plt.axis(
        "equal"
    )

    plt.xlim(
        -0,
        5
    )

    plt.ylim(
        -2,
        2
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

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
        "BARTLETT + 2-D CA-CFAR "
        "RANGE-ANGLE DETECTION"
    )

    print(
        "=" * 75
    )

    print(
        f"Wavelength: "
        f"{LAMBDA * 1e3:.4f} mm"
    )

    print(
        f"Half wavelength: "
        f"{HALF_WAVELENGTH * 1e3:.4f} mm"
    )

    print(
        f"TX-RX spatial channels: "
        f"{NUM_VIRTUAL_PAIRS}"
    )

    # ========================================================
    # RANGE AXIS
    # ========================================================

    freq_axis, range_axis = (
        make_range_axis()
    )

    print(
        f"Range-bin spacing: "
        f"{range_axis[1] - range_axis[0]:.6f} m"
    )

    # ========================================================
    # LOAD
    # ========================================================

    adc_data = {}

    for chip_name, file_path in FILES.items():

        adc_data[
            chip_name
        ] = load_dca1000_iq(
            file_path
        )

    # ========================================================
    # FRAME COUNT
    #
    # No frame progress prints.
    # ========================================================

    frame_counts = []

    for chip_name in FILES.keys():

        num_chirps = (
            adc_data[
                chip_name
            ].shape[0]
        )

        if (
            num_chirps
            % CHIRPS_PER_FRAME
            != 0
        ):

            raise ValueError(
                f"{chip_name}: "
                f"{num_chirps} chirps "
                f"cannot form complete frames."
            )

        frame_counts.append(
            num_chirps
            // CHIRPS_PER_FRAME
        )

    if len(
        set(frame_counts)
    ) != 1:

        raise ValueError(
            "All cascade files must contain "
            "the same number of complete frames."
        )

    num_frames = (
        frame_counts[0]
    )

    # ========================================================
    # ACCUMULATE BARTLETT POWER
    # ========================================================

    accumulated_power = None

    # ========================================================
    # PROCESS EACH FRAME
    # ========================================================

    for frame_id in range(
        num_frames
    ):

        mimo_data_by_chip = {}

        # ----------------------------------------------------
        # Range FFT for each chip
        # ----------------------------------------------------

        for chip_name in FILES.keys():

            start = (
                frame_id
                * CHIRPS_PER_FRAME
            )

            end = (
                start
                + CHIRPS_PER_FRAME
            )

            frame_adc = (
                adc_data[
                    chip_name
                ][
                    start:end
                ]
            )

            range_fft = (
                compute_range_fft(
                    frame_adc
                )
            )

            mimo_data_by_chip[
                chip_name
            ] = arrange_mimo_data(
                range_fft
            )

        # ----------------------------------------------------
        # Spatial snapshots
        #
        # (loop, TX-RX pair, range)
        # ----------------------------------------------------

        snapshots = (
            build_spatial_snapshots(
                mimo_data_by_chip
            )
        )

        # ----------------------------------------------------
        # Bartlett
        # ----------------------------------------------------

        pair_x_positions_m = np.array([
            p["pair_x_m"]
            for p in VIRTUAL_PAIRS
        ])

        angle_axis_deg, bartlett_power = (
            compute_bartlett_map(
                snapshots,
                pair_x_positions_m
            )
        )

        if accumulated_power is None:

            accumulated_power = (
                bartlett_power.copy()
            )

        else:

            accumulated_power += (
                bartlett_power
            )

    # ========================================================
    # AVERAGE OVER FRAMES
    # ========================================================

    bartlett_power = (
        accumulated_power
        / max(
            1,
            num_frames
        )
    )

    # ========================================================
    # SELECT RANGE
    # ========================================================

    selected_range_axis, selected_power = (
        select_range(
            range_axis,
            bartlett_power
        )
    )

    # ========================================================
    # NORMALIZED BARTLETT dB
    # ========================================================

    bartlett_db = (
        10.0
        * np.log10(
            selected_power
            + 1e-18
        )
    )

    bartlett_db -= np.max(
        bartlett_db
    )

    bartlett_db = np.maximum(
        bartlett_db,
        DB_FLOOR
    )

    # ========================================================
    # PLOT RAW BARTLETT MAP
    # ========================================================

    plot_bartlett_map(
        selected_range_axis,
        angle_axis_deg,
        bartlett_db
    )

    # ========================================================
    # 2-D CA-CFAR
    #
    # IMPORTANT:
    # CFAR operates on LINEAR POWER.
    # ========================================================

    detections_mask, threshold_map, noise_map = (
        ca_cfar_2d(
            selected_power,
            training_cells=TRAINING_CELLS,
            guard_cells=GUARD_CELLS,
            pfa=PFA
        )
    )

    print(
        f"\nRaw CA-CFAR detections: "
        f"{np.sum(detections_mask)}"
    )

    # ========================================================
    # EXTRACT CFAR CANDIDATES
    # ========================================================

    candidates = (
        extract_cfar_candidates(
            detections_mask,
            selected_power,
            selected_range_axis,
            angle_axis_deg,
            minimum_db=MIN_DETECTION_DB
        )
    )

    print(
        f"CFAR candidates after dB filtering: "
        f"{len(candidates)}"
    )

    # ========================================================
    # NON-MAXIMUM SUPPRESSION
    # ========================================================

    detections = (
        suppress_detections(
            candidates,
            max_detections=MAX_DETECTIONS,
            min_range_separation=(
                MIN_DETECTION_RANGE_M
            ),
            min_angle_separation=(
                MIN_DETECTION_ANGLE_DEG
            )
        )
    )

    # ========================================================
    # X/Y
    # ========================================================

    detections = (
        add_xy_coordinates(
            detections
        )
    )

    # ========================================================
    # PRINT FINAL DETECTIONS
    # ========================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "FINAL CA-CFAR DETECTIONS"
    )

    print(
        "=" * 90
    )

    print(
        f"{'ID':>4s}"
        f"{'Range(m)':>12s}"
        f"{'Angle(deg)':>14s}"
        f"{'X(m)':>12s}"
        f"{'Y(m)':>12s}"
        f"{'Power(dB)':>14s}"
    )

    print(
        "-" * 90
    )

    for i, d in enumerate(
        detections,
        start=1
    ):

        print(
            f"{i:4d}"
            f"{d['range_m']:12.3f}"
            f"{d['angle_deg']:14.2f}"
            f"{d['x_m']:12.3f}"
            f"{d['y_m']:12.3f}"
            f"{d['relative_power_db']:14.2f}"
        )

    print(
        "\nNumber of final detections: "
        f"{len(detections)}"
    )

    # ========================================================
    # RANGE-ANGLE + CFAR
    # ========================================================

    plot_cfar_range_angle(
        selected_range_axis,
        angle_axis_deg,
        bartlett_db,
        detections
    )

    # ========================================================
    # X-Y
    # ========================================================

    plot_xy_detections(
        detections
    )

    # ========================================================
    # SAVE DETECTIONS
    # ========================================================

    if len(
        detections
    ) > 0:

        output = np.array([
            [
                d["range_m"],
                d["angle_deg"],
                d["x_m"],
                d["y_m"],
                d["relative_power_db"]
            ]
            for d in detections
        ])

        np.savetxt(
            "bartlett_cfar_detections_xy.txt",
            output,
            header=(
                "range_m "
                "angle_deg "
                "x_m "
                "y_m "
                "relative_power_db"
            )
        )

        print(
            "\nSaved:"
        )

        print(
            "bartlett_cfar_detections_xy.txt"
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()