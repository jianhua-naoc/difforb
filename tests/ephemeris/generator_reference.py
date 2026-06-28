from pathlib import Path

DE441_SPK_PATH = Path(__file__).resolve().parents[1] / "data" / "spk" / "de441_2017_2025_excerpt.bsp"
SB441_SPK_PATH = Path(__file__).resolve().parents[1] / "data" / "spk" / "sb441_2017_2025_excerpt.bsp"
EPOCH_TDB_JD = 2460690.5
HORIZONS_TARGET_TDB_JD = 2460692.5
HORIZONS_OBSERVER_UTC_JD = 2460692.5

# Hard-coded JPL Horizons ELEMENTS references for propagation through
# EphemerisGenerator. The initial elements are queried at JD 2460690.5 TDB and
# the expected elements at JD 2460692.5 TDB, with CENTER="500@10",
# REF_PLANE="ECLIPTIC", REF_SYSTEM="ICRF", and OUT_UNITS="AU-D".
# The DiffOrb propagation path uses IAS15 and the extended dynamical system
# built from the DE441 and SB441 SPK excerpts in ``tests/data/spk``.
HORIZONS_GENERATOR_ELEMENT_CASES = [
    (
        "nominal elliptical",
        "248370;",
        "248370 (2005 QN173)",
        (3.066194321327381, 0.2246346475233367, 0.06809604774029922, 174.1147890863298, 146.0086279341069, 246.7651321160856),
        (3.06620326919095, 0.2246327314584845, 0.06809702249023837, 174.1149647114716, 146.0077796096423, 247.1329477341933),
    ),
    (
        "near-parabolic elliptical",
        "C/2020 F3;",
        "NEOWISE (C/2020 F3)",
        (393.52720951255, 0.9992507683148615, 129.1795036028305, 61.10589246573335, 37.371487395643, 0.2090039409219019),
        (393.6400938950702, 0.9992510060955531, 129.1791693143346, 61.10572317257797, 37.3711789001622, 0.2091659059691625),
    ),
    (
        "hyperbolic",
        "1I;",
        "1I/'Oumuamua (A/2017 U1)",
        (-1.273991681936413, 1.204095519177622, 122.1130712681111, 24.24996200983482, 241.828905068838, 1841.186605475453),
        (-1.273993210645332, 1.204117145073093, 122.1161360970526, 24.25179815494103, 241.8314789002654, 1842.556065579187),
    ),
]

# Hard-coded JPL Horizons VECTORS references for Maunakea (observatory code
# 568). The rows were queried at JD 2460692.5 TDB with CENTER="568",
# REF_PLANE="FRAME", REF_SYSTEM="ICRF", OUT_UNITS="AU-D", VEC_TABLE="3", and
# VEC_CORR set to "NONE", "LT", and "LT+S" for geometric, astrometric, and
# apparent vectors respectively.
HORIZONS_GENERATOR_VECTOR_CASES = [
    (
        "nominal elliptical",
        "248370;",
        "248370 (2005 QN173)",
        (3.066194321327381, 0.2246346475233367, 0.06809604774029922, 174.1147890863298, 146.0086279341069, 246.7651321160856),
        (
            [-2.983065454841139, -1.155644071357786, -0.5000597772031314],
            [0.01804219818163885, -0.0006337087046280925, -0.0001804535662854261],
        ),
        (
            [-2.983113569723612, -1.155501846134588, -0.4999983057755378],
            [0.01804173528727114, -0.0006337560283066505, -0.0001804739548312219],
        ),
        (
            [-2.983106179160675, -1.155517303700104, -0.5000066767264565],
            [0.01804173528727114, -0.0006337560283066505, -0.0001804739548312219],
        ),
    ),
    (
        "near-parabolic elliptical",
        "C/2020 F3;",
        "NEOWISE (C/2020 F3)",
        (393.52720951255, 0.9992507683148615, 129.1795036028305, 61.10589246573335, 37.371487395643, 0.2090039409219019),
        (
            [-9.373739877292522, -8.826707727369714, -8.476663504838296],
            [0.01117257502696032, 0.004432342427476429, -0.0005821174623444904],
        ),
        (
            [-9.373357364290031, -8.82648165765168, -8.476335096912797],
            [0.01117249923238712, 0.004432280492654646, -0.0005821802093856419],
        ),
        (
            [-9.373916700994215, -8.826332226364203, -8.475872139159119],
            [0.01117249923238712, 0.004432280492654646, -0.0005821802093856419],
        ),
    ),
    (
        "hyperbolic",
        "1I;",
        "1I/'Oumuamua (A/2017 U1)",
        (-1.273991681936413, 1.204095519177622, 122.1130712681111, 24.24996200983482, 241.828905068838, 1841.186605475453),
        (
            [41.42499682574222, -1.598711489995986, 18.0813980522897],
            [0.02969657634616462, 0.006440741963266885, 0.009658386930818089],
        ),
        (
            [41.42128076245491, -1.598572832023328, 18.07968681458312],
            [0.02969661129770138, 0.006440741282288819, 0.009658402655490015],
        ),
        (
            [41.42086805292332, -1.600533824849159, 18.08045883357613],
            [0.02969661129770138, 0.006440741282288819, 0.009658402655490015],
        ),
    ),
]

# Hard-coded JPL Horizons OBSERVER references for Maunakea (observatory code
# 568). The target initial conditions are the same ELEMENTS references at JD
# 2460690.5 TDB above. The observer rows were queried at JD 2460692.5 UTC with
# LOCATION="568", REF_SYSTEM="ICRF", and quantities including "Astrometric
# RA/DEC", "Azimuth and elevation", and "Target range".
HORIZONS_GENERATOR_OBSERVER_CASES = [
    (
        "nominal elliptical",
        "248370;",
        "248370 (2005 QN173)",
        (3.066194321327381, 0.2246346475233367, 0.06809604774029922, 174.1147890863298, 146.0086279341069, 246.7651321160856),
        (201.17386, -8.88317, 271.238673, -30.891039, 3.23791051687642),
    ),
    (
        "near-parabolic elliptical",
        "C/2020 F3;",
        "NEOWISE (C/2020 F3)",
        (393.52720951255, 0.9992507683148615, 129.1795036028305, 61.10589246573335, 37.371487395643, 0.2090039409219019),
        (223.27889, -33.3591, 239.790007, -16.855468, 15.4147540021869),
    ),
    (
        "hyperbolic",
        "1I;",
        "1I/'Oumuamua (A/2017 U1)",
        (-1.273991681936413, 1.204095519177622, 122.1130712681111, 24.24996200983482, 241.828905068838, 1841.186605475453),
        (357.78989, 23.56482, 76.709435, 55.599059, 45.2233926564212),
    ),
]

# Hard-coded JPL Horizons VECTORS references for heliocentric apsides. The
# initial elements are queried at each search-start epoch with CENTER="500@10",
# REF_PLANE="ECLIPTIC", REF_SYSTEM="ICRF", and OUT_UNITS="AU-D". The expected
# event epochs and distances are independent roots of the Horizons geometric
# vector radial velocity with CENTER="500@10", REF_PLANE="FRAME",
# VEC_TABLE="3", and VEC_CORR="NONE".
HORIZONS_GENERATOR_APSIDES_CASES = [
    (
        "nominal elliptical",
        "248370;",
        "248370 (2005 QN173)",
        2459300.5,
        (3.067442808579839, 0.2262112507373586, 0.06666323366149676, 174.4752453408724, 145.9641552870563, 351.1368348315615),
        2459300.5,
        2460360.5,
        (
            (0, 2459348.694289398380, 2.37358555398324),
            (1, 2460327.297458762303, 3.75467523716913),
        ),
        5.0e-6,
        2.0e-9,
    ),
    (
        "near-parabolic",
        "DES=C/2020 F3;",
        "NEOWISE (C/2020 F3)",
        2459028.5,
        (358.8284493471022, 0.9991788520410725, 128.9375132776647, 61.0104316929371, 37.27864620045761, 359.9991765491615),
        2459028.5,
        2459040.5,
        (
            (0, 2459034.178897429258, 0.294651246479942),
        ),
        5.0e-6,
        5.0e-10,
    ),
    (
        "hyperbolic",
        "DES=A/2017 U1;",
        "1I/'Oumuamua (A/2017 U1)",
        2458005.9,
        (-1.283884522841206, 1.19943407501728, 122.7370049014244, 24.60033902680285, 241.8416927737729, -0.06413355192593183),
        2458005.9,
        2458006.8,
        (
            (0, 2458005.994584860280, 0.256050341693878),
        ),
        1.0e-4,
        1.0e-7,
    ),
]

# Hard-coded JPL Horizons APPROACH references for representative targets. The
# initial elements are queried at each search-start epoch with CENTER="500@10",
# REF_PLANE="ECLIPTIC", REF_SYSTEM="ICRF", OUT_UNITS="AU-D". The close
# approach rows are queried with EPHEM_TYPE="APPROACH",
# CA_TABLE_TYPE="EXTENDED", and a widened CALIM_PL window:
# "0.39,0.72,1.0,1.52,2.0,2.0,2.0,2.0,0.5,0.003".
HORIZONS_GENERATOR_CLOSE_APPROACH_CASES = [
    (
        "nominal elliptical mars approach",
        "248370;",
        "248370 (2005 QN173)",
        2460938.5,
        (3.066981660087262, 0.2245362632882199, 0.06814237677395862, 174.14140810149, 145.916037335945, 292.3454847998682),
        2460938.5,
        2460953.5,
        "mars barycenter",
        1.52,
        2460946.09418,
        1.392351,
        6.222,
        5.0e-6,
        5.0e-7,
        6.0e-7,
    ),
    (
        "near-parabolic earth approach",
        "DES=C/2020 F3;",
        "NEOWISE (C/2020 F3)",
        2459050.5,
        (358.6997226286382, 0.9991785566783854, 128.9374831610417, 61.01049873791632, 37.27878825714552, 0.002367864167618191),
        2459050.5,
        2459056.5,
        "earth",
        1.0,
        2459053.54612,
        0.691830,
        79.019,
        5.0e-6,
        5.0e-7,
        6.0e-7,
    ),
    (
        "hyperbolic earth approach",
        "DES=A/2017 U1;",
        "1I/'Oumuamua (A/2017 U1)",
        2458039.5,
        (-1.273651121345333, 1.200933668921794, 122.7399769998179, 24.59685741124158, 241.803606587221, 22.97086673608008),
        2458039.5,
        2458045.5,
        "earth",
        1.0,
        2458041.24349,
        0.161754,
        60.259,
        5.0e-5,
        5.0e-7,
        6.0e-7,
    ),
]
