# data_loaders/load_pcap.py

import glob
import os
import re
import pyshark
import multiprocessing as mp
from functools import partial
from typing import Optional
from src.common.common import count_frames_in_devices


# ======================= MERGING HELPERS =======================

def _merge_pr_list(pr_list, new_pr):
    """
    If a PROBE_REQ with the same DATA exists,
    append TIME and RSSI to it; otherwise add a new entry.
    """
    found = False
    for pr in pr_list:
        if pr["DATA"] == new_pr["DATA"]:
            pr["TIME"].extend(new_pr["TIME"])
            pr["RSSI"].extend(new_pr["RSSI"])
            found = True
            break
    if not found:
        pr_list.append(new_pr)


def _merge_device_into(target_by_id, src_dev, labeled: bool):
    """
    Merges device src_dev into target_by_id.
    The key is always the MAC; LABEL is optionally set.
    """
    device_id = src_dev.get("MAC")
    if device_id is None:
        return

    dev = target_by_id.get(device_id)
    if dev is None:
        dev = {
            "MAC": device_id,
            "PROBE_REQs": []
        }
        if labeled and src_dev.get("LABEL") is not None:
            dev["LABEL"] = src_dev["LABEL"]
        target_by_id[device_id] = dev
    else:
        if labeled and dev.get("LABEL") is None and src_dev.get("LABEL") is not None:
            dev["LABEL"] = src_dev["LABEL"]

    # PROBE_REQs (merging by identical DATA)
    for pr in src_dev.get("PROBE_REQs", []) or []:
        _merge_pr_list(
            dev["PROBE_REQs"],
            {
                "TIME": list(pr.get("TIME", [])),
                "RSSI": list(pr.get("RSSI", [])),
                "DATA": pr.get("DATA", {})
            }
        )


# ======================= PARSING HELPERS =======================

def _to_int_safe(tok):
    """Tries to convert to int from various forms ('42', 42, '0x2A', ' -40 ', ' -43.0 ', etc.)."""
    if tok is None:
        return None
    if isinstance(tok, int):
        return tok
    s = str(tok).strip()
    try:
        return int(s)
    except Exception:
        pass
    try:
        return int(float(s))
    except Exception:
        pass
    m = re.search(r'0x([0-9A-Fa-f]+)', s)
    if m:
        try:
            return int(m.group(1), 16)
        except Exception:
            pass
    m = re.search(r'x?([0-9A-Fa-f]{2,})$', s)
    if m:
        try:
            return int(m.group(1), 16)
        except Exception:
            pass
    m = re.search(r'([+-]?\d+)$', s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    return None


def _safe_raw_at(raws, i, cut_chars=0):
    """Returns raws[i][0] as a string (trimmed by cut_chars), or None if missing/wrong format."""
    try:
        if not isinstance(raws, list) or i < 0 or i >= len(raws):
            return None
        row = raws[i]
        raw = row[0] if isinstance(row, list) else row
        if not isinstance(raw, str):
            return None
        if cut_chars:
            if len(raw) < cut_chars:
                return None
            raw = raw[cut_chars:]
        return raw
    except Exception:
        return None


# ======================= SINGLE PCAP FILE =======================

def process_pcap_file(file_path: str, labeled: bool, label_from: str):
    """
    Parses a single PCAP file and returns the local devices_by_id,
    structurally compatible with the JSONs:
      [
        {
          "MAC": "...",
          "LABEL": "...",   # only for labeled
          "PROBE_REQs": [ ... ]
        },
        ...
      ]
    """
    devices_by_id = {}

    folder_label = os.path.basename(os.path.dirname(file_path))
    file_label = os.path.splitext(os.path.basename(file_path))[0][0]

    print(f"[worker] Processing file: {file_path}")

    try:
        capture = pyshark.FileCapture(
            file_path,
            use_json=True,
            include_raw=True,
            keep_packets=False,
            # if the PCAPs contain all frames, not just Probe Requests,
            # you can uncomment:
            # display_filter="wlan.fc.type_subtype == 4",
        )

        for pkt in capture:
            try:
                if not hasattr(pkt, "wlan") or not hasattr(pkt, "wlan_radio"):
                    continue

                pr_data = {}
                ssid = ""

                # source MAC
                try:
                    mac = pkt.wlan.sa
                except Exception:
                    continue

                # time as a string (same as in the JSONs)
                sniff_time = pkt.sniff_timestamp

                # RSSI as a string (same as in the JSONs)
                try:
                    rssi_val = int(pkt.wlan_radio.signal_dbm)
                except Exception:
                    rssi_val = -200  # fallback
                rssi = str(rssi_val)

                # device identifier - always the MAC
                device_id = mac

                # label for labeled mode
                label_value = None
                if labeled:
                    label_value = file_label if label_from == "file" else folder_label

                # tagged fields
                try:
                    tagged = pkt[7]._all_fields.get("wlan.tagged.all", {})
                except Exception:
                    tagged = {}

                tag_list = tagged.get("wlan.tag", [])
                if not isinstance(tag_list, list):
                    tag_list = [tag_list]
                tag_raws = tagged.get("wlan.tag_raw", [])

                for i, tag in enumerate(tag_list):
                    try:
                        tag_num = tag["wlan.tag.number"]
                        pr_data["TAG_LIST"] = pr_data.get("TAG_LIST", []) + [tag_num]
                    except Exception:
                        continue

                    if tag_num == "0":
                        ssid = tag.get("wlan.ssid", "")
                        pr_data["SSID"] = ssid
                    elif tag_num == "1":
                        supp = tag.get("wlan.supported_rates", [])
                        if not isinstance(supp, list):
                            supp = [supp]
                        supp_vals = []
                        for x in supp:
                            v = _to_int_safe(x)
                            if v is not None:
                                supp_vals.append(str(v))
                        pr_data.setdefault("DATA_RTS", {})["SUPP"] = supp_vals

                    elif tag_num == "50":
                        ext = tag.get("wlan.extended_supported_rates", [])
                        if not isinstance(ext, list):
                            ext = [ext]
                        ext_vals = []
                        for x in ext:
                            v = _to_int_safe(x)
                            if v is not None:
                                ext_vals.append(str(v))
                        pr_data.setdefault("DATA_RTS", {})["EXT"] = ext_vals

                    elif tag_num == "45":
                        raw_hex = _safe_raw_at(tag_raws, i, cut_chars=4)
                        if raw_hex is not None:
                            pr_data["HT_CAP"] = "0x" + raw_hex
                            ht_bytes = bytes.fromhex(raw_hex)

                            # Offsets per IEEE 802.11-2020, 9.4.2.55 (HT Capabilities element, 26 B body).
                            pr_data["HT_CAP_INFO"]    = "0x" + ht_bytes[0:2].hex()   # 2 B
                            pr_data["HT_AMPDU_PARMS"] = "0x" + ht_bytes[2:3].hex()   # 1 B
                            pr_data["HT_MCS_SET"]     = "0x" + ht_bytes[3:19].hex()  # 16 B (3..18)
                            pr_data["HT_EXT_CAP"]     = "0x" + ht_bytes[19:21].hex() # 2 B (19..20)
                            pr_data["HT_Tx"]          = "0x" + ht_bytes[21:25].hex() # 4 B (21..24)
                            pr_data["HT_ANTENNA"]     = "0x" + ht_bytes[25:26].hex() # 1 B (25)

                    elif tag_num == "127":
                        raw_hex = _safe_raw_at(tag_raws, i, cut_chars=4)
                        if raw_hex is not None:
                            pr_data["EXT_CAP"] = "0x" + raw_hex

                    elif tag_num == "191":
                        raw_hex = _safe_raw_at(tag_raws, i, cut_chars=4)
                        if raw_hex is not None:
                            pr_data["VHT_CAP"] = "0x" + raw_hex

                    elif tag_num == "107":
                        raw_hex = _safe_raw_at(tag_raws, i, cut_chars=4)
                        if raw_hex is not None:
                            pr_data["INTWORK"] = "0x" + raw_hex

                    elif tag_num == "221":
                        vendor_spec = pr_data.setdefault("VENDOR_SPEC", {})
                        oui = tag["wlan.tag.vendor.oui"] if "wlan.tag.vendor.oui" in tag else "UNKNOWN_OUI"
                        length = _to_int_safe(tag.get("wlan.tag.length"))
                        raw_data = _safe_raw_at(tag_raws, i, cut_chars=0)
                        if length is None or raw_data is None:
                            continue
                        data_id = tag["wlan.tag.vendor.oui.type"] if "wlan.tag.vendor.oui.type" in tag else "UNKNOWN_TYPE"
                        data = "0x" + raw_data[-max(0, (length - 4)) * 2:]
                        vendor_spec.setdefault(oui, {})[data_id] = data

                # EXT_TAG
                if "wlan.ext_tag" in tagged:
                    ext_tags = tagged["wlan.ext_tag"]
                    ext_tag_raws = tagged.get("wlan.ext_tag_raw", [])
                    pr_data["EXT_TAG"] = {}
                    if isinstance(ext_tags, list):
                        for i, ext_tag in enumerate(ext_tags):
                            try:
                                tag_num = ext_tag["wlan.ext_tag.number"]
                            except Exception:
                                continue
                            raw_data = _safe_raw_at(ext_tag_raws, i, cut_chars=2)
                            if raw_data is None:
                                continue
                            pr_data["EXT_TAG"][tag_num] = "0x" + raw_data
                    else:
                        try:
                            tag_num = ext_tags["wlan.ext_tag.number"]
                            raw_data = _safe_raw_at(ext_tag_raws, 0, cut_chars=2)
                            if raw_data is not None:
                                pr_data["EXT_TAG"] = {tag_num: "0x" + raw_data}
                        except Exception:
                            pass

                # attach to the device
                dev = devices_by_id.get(device_id)
                if dev is None:
                    dev = {
                        "MAC": mac,
                        "PROBE_REQs": [
                            {"TIME": [sniff_time], "RSSI": [rssi], "DATA": pr_data}
                        ],
                    }
                    if labeled and label_value is not None:
                        dev["LABEL"] = label_value
                    devices_by_id[device_id] = dev
                else:
                    if labeled and dev.get("LABEL") is None and label_value is not None:
                        dev["LABEL"] = label_value

                    _merge_pr_list(
                        dev["PROBE_REQs"],
                        {"TIME": [sniff_time], "RSSI": [rssi], "DATA": pr_data},
                    )

            except Exception as e:
                print(f"[worker] Packet error in {file_path}: {e}")

        capture.close()

    except Exception as e:
        print(f"[worker] Error loading {file_path}: {e}")

    return devices_by_id


# ======================= PUBLIC API =======================

def load_pcap(
    input_dir: str,
    labeled: bool = False,
    label_from: str = "folder",   # 'folder' or 'file'
    workers: Optional[int] = None
):
    """
    Loads all PCAP files from the given directory (recursively)
    and returns a list of devices in a format compatible with the JSONs:
      [
        {
          "MAC": "...",
          "LABEL": "...",   # only for labeled
          "PROBE_REQs": [ ... ]
        },
        ...
      ]
    """
    print("Loading PCAP files from:", input_dir)
    pcap_files = glob.glob(os.path.join(input_dir, "**", "*.pcap"), recursive=True)
    if not pcap_files:
        print("No PCAP files found.")
        return []

    # number of workers
    if workers is not None:
        max_procs = max(1, min(int(workers), len(pcap_files)))
    else:
        max_procs = min(max(1, (os.cpu_count() or 2) - 1), 6, len(pcap_files))

    print(f"Starting parallel processing: {max_procs} process(es)")
    devices_by_id = {}

    with mp.get_context("spawn").Pool(processes=max_procs) as pool:
        worker = partial(process_pcap_file, labeled=labeled, label_from=label_from)
        for local_map in pool.imap_unordered(worker, pcap_files, chunksize=1):
            if not local_map:
                continue
            for dev_id, src_dev in local_map.items():
                _merge_device_into(devices_by_id, src_dev, labeled)

    # convert to list
    devices = list(devices_by_id.values())

    print("Loaded PCAP file(s). Number of frames:", count_frames_in_devices(devices))
    return devices
