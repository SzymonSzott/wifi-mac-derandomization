from numpy import array, transpose, fill_diagonal
from itertools import product

from src.data_processing.feature_extraction import exctract_fields_values as ext
from src.data_processing.distances import (
    bit_dist,
    dist_data_rates,
    dist_ext_tag,
    dist_vendor_spec,
    dist_IE_list,
    dist_rssi_time,
    dist_ssid
)

def mac_distance(fields_by_mac, mac1, mac2, fields_cfg, null_token="NULL"):
    d1 = fields_by_mac.get(mac1, {})
    d2 = fields_by_mac.get(mac2, {})
    min_dist = 0.0

    per_field = {}

    for field, cfg in fields_cfg.items():
        kind = cfg["kind"]

        l1 = [v for v in (d1.get(field) or []) if v not in (null_token, None)]
        l2 = [v for v in (d2.get(field) or []) if v not in (null_token, None)]

        if not l1 and not l2:
            per_field[field] = 0.0
            continue
        if (not l1) ^ (not l2):
            # one is empty, the other is not
            per_field[field] = float(cfg.get("penalty", 0.0))
            continue

        if kind == "hex_bit":
            weight = float(cfg.get("weight", 1.0))
            penalty = float(cfg.get("penalty", 0.0))

            s = 0.0
            n = 0
            for a, b in product(l1, l2):
                if not (isinstance(a, str) and isinstance(b, str) and a.startswith("0x") and b.startswith("0x")):
                    s += penalty; n += 1; continue
                if len(a) != len(b):
                    s += penalty; n += 1; continue

                v1 = int(a, 16)
                v2 = int(b, 16)
                v1_bit = int(bin(v1)[2:], 16)
                v2_bit = int(bin(v2)[2:], 16)
                length = (len(a) - 2) * 4

                s += float(bit_dist(v1_bit, v2_bit, weight, length))
                n += 1

            per_field[field] = (s / n) if n else 0.0

        elif kind == "ie_list":
            weight = float(cfg.get("weight", 1.0))
            per_field[field] = float(dist_IE_list(l1, l2, weight))

        elif kind == "data_rates":
            weight = float(cfg.get("weight", 1.0))
            if l1 and l2 and isinstance(l1[0], (list, tuple)) and isinstance(l2[0], (list, tuple)):
                s = 0.0
                n = 0
                for a, b in product(l1, l2):
                    s += float(dist_data_rates(a, b, weight))
                    n += 1
                per_field[field] = (s / n) if n else 0.0
            else:
                per_field[field] = float(dist_data_rates(l1, l2, weight))

        elif kind == "ext_tag":
            w1 = float(cfg["weight1"]); w2 = float(cfg["weight2"]); w3 = float(cfg["weight3"])
            s = 0.0
            n = 0
            for a, b in product(l1, l2):
                s += float(dist_ext_tag(a, b, w1, w2, w3))
                n += 1
            per_field[field] = (s / n) if n else 0.0

        elif kind == "vendor_spec":
            w1 = float(cfg["weight1"]); w2 = float(cfg["weight2"]); w3 = float(cfg["weight3"])
            s = 0.0
            n = 0
            for a, b in product(l1, l2):
                s += float(dist_vendor_spec(a, b, w1, w2, w3))
                n += 1
            per_field[field] = (s / n) if n else 0.0

        elif kind == "rssi_time":
            w1 = float(cfg["weight1"]); w2 = float(cfg["weight2"])
            w3 = float(cfg["weight3"]); w4 = float(cfg["weight4"])

            # take RSSI and TIME from the data (not from "field")
            rssi1 = [float(x) for x in (d1.get("RSSI") or []) if x not in (null_token, None)]
            time1 = [float(x) for x in (d1.get("TIME") or []) if x not in (null_token, None)]
            rssi2 = [float(x) for x in (d2.get("RSSI") or []) if x not in (null_token, None)]
            time2 = [float(x) for x in (d2.get("TIME") or []) if x not in (null_token, None)]

            if not rssi1 or not time1 or not rssi2 or not time2:
                per_field[field] = float(cfg.get("penalty", 0.0))
            else:
                per_field[field] = float(dist_rssi_time(rssi1, time1, rssi2, time2, w1, w2, w3, w4))

        elif kind == "ssid":
            weight = float(cfg.get("weight", 1.0))
            s = 0.0
            n = 0
            for a, b in product(l1, l2):
                s += float(dist_ssid(a, b, weight))
                n += 1
            per_field[field] = (s / n) if n else 0.0

        else:
            raise ValueError(f"Unknown kind for field '{field}': {kind}")

    total_distance = sum(per_field.values())

    if total_distance < min_dist:
        total_distance = min_dist

    # print(f"Distance between {mac1} and {mac2}: {total_distance} (per field: {per_field})")

    return {
        "mac1": mac1,
        "mac2": mac2,
        "per_field": per_field,
        "total_distance": float(total_distance),
    }


def mac_distance_from_df(df, mac1, mac2, fields_cfg, null_token="NULL"):
    """Convenience wrapper: calls ext(df, fields) itself."""
    fields = list(fields_cfg.keys())
    fields_by_mac = ext(df, fields)
    return mac_distance(fields_by_mac, mac1, mac2, fields_cfg, null_token=null_token)


def calculate_distance_matrix(fields_by_mac, fields_cfg, null_token="NULL"):
    macs = list(fields_by_mac.keys())
    n = len(macs)
    # matrix = [[None] * n for _ in range(n)]
    matrix = [[0] * n for _ in range(n)]

    for i, mac1 in enumerate(macs):
        msg = f"Calculating distances for {mac1} ({i+1}/{n})..."
        print(f"\r{msg:<80}", end="", flush=True)

        # Potentially dodgy calculation and transformation in order to cut the time of calculation (assuming that mac_distance(a, b) == mac_distance(b, a))
        # Calculating only the lower hald of the matrix (hence the macs[:i+1])
        # Then mirroring the lower half into the upper half (with clearing up the diagonal just in case)
        for j, mac2 in enumerate(macs[:i+1]):
            dist_info = mac_distance(fields_by_mac, mac1, mac2, fields_cfg, null_token=null_token)
            matrix[i][j] = dist_info["total_distance"]

    arr_matrix = array(matrix)
    transposed_matrix = transpose(arr_matrix)
    fill_diagonal(transposed_matrix, 0)
    end_matrix = arr_matrix + transposed_matrix

    print()
    return end_matrix.tolist()
    # return matrix