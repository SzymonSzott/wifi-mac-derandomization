# evaluation/validation.py

import json
from tabulate import tabulate
from collections import defaultdict, Counter
import math
from collections import Counter, defaultdict

def display_report(clustered_data, mac_to_label):
    # === MAC -> cluster validation ===
    correct_count = 0
    incorrect_count = 0
    total_count = 0
    incorrect_macs = []

    mac_to_cluster = {}
    for cluster_id, mac_list in clustered_data.items():
        for mac in mac_list:
            mac_to_cluster[mac] = int(cluster_id)

    label_to_macs = defaultdict(list)
    for mac, label in mac_to_label.items():
        label_to_macs[label].append(mac)


    for label, macs in label_to_macs.items():
        assigned_clusters = [mac_to_cluster.get(mac) for mac in macs if mac in mac_to_cluster]
        labeled_macs = [mac for mac in macs if mac in mac_to_cluster]

        if not assigned_clusters:
            continue

        total_count += len(labeled_macs)
        cluster_counts = Counter(assigned_clusters)
        expected_cluster = cluster_counts.most_common(1)[0][0]

        for mac in labeled_macs:
            actual_cluster = mac_to_cluster[mac]
            if actual_cluster == expected_cluster and actual_cluster!=-1:
                correct_count += 1
            else:
                incorrect_count += 1
                incorrect_macs.append({
                    "MAC": mac,
                    "label": label,
                    "cluster": actual_cluster,
                    "expected": expected_cluster
                })

    # === SUCCESS METRICS ===
    # 1) Percentage of pure clusters
    cluster_to_labels = defaultdict(set)
    for cluster_id_str, mac_list in clustered_data.items():
        for mac in mac_list:
            if mac in mac_to_label:
                cluster_to_labels[int(cluster_id_str)].add(mac_to_label[mac])

    clusters_with_labels = [cid for cid, labels in cluster_to_labels.items() if len(labels) > 0]
    pure_clusters = [cid for cid, labels in cluster_to_labels.items() if len(labels) == 1]

    num_clusters_with_labels = len(clusters_with_labels)
    num_pure_clusters = len(pure_clusters)
    pure_clusters_pct = (num_pure_clusters / num_clusters_with_labels * 100.0) if num_clusters_with_labels > 0 else 0.0

    # 2) Devices recognized as a single device
    correctly_one_cluster_devices = 0
    per_device_single_cluster_flag = {}
    for label, macs in label_to_macs.items():
        macs_in_clusters = [mac for mac in macs if mac in mac_to_cluster]
        clusters_for_device = set(mac_to_cluster[mac] for mac in macs_in_clusters)
        is_single_cluster = (len(macs_in_clusters) > 0) and (len(clusters_for_device) == 1)
        per_device_single_cluster_flag[label] = is_single_cluster
        if is_single_cluster:
            correctly_one_cluster_devices += 1

    total_devices = len(label_to_macs)
    devices_one_cluster_pct = (correctly_one_cluster_devices / total_devices * 100.0) if total_devices > 0 else 0.0

    # noise = clustered_data.get("-1", [])
    # noise_count = len(noise)
    noise_pct = (len(mac_to_label.keys()) - len(mac_to_cluster.keys())) / len(mac_to_label.keys()) * 100.0 if len(mac_to_label.keys()) > 0 else 0.0  # number of labeled MACs not assigned to any cluster (absent from mac_to_cluster)

    # === REPORT ===
    print("\n--- CLUSTER VALIDATION ---")
    print(f"Correctly classified MACs: {correct_count}")
    print(f"Incorrectly classified MACs: {incorrect_count}")
    print(f"Total number of analyzed MACs: {total_count}")

    print("\n--- NOISE ---")
    print(f"Percentage of MACs classified as noise: {noise_pct:.1f}% ({len(mac_to_label.keys()) - len(mac_to_cluster.keys())} of {len(mac_to_label.keys())})")

    print("\n--- SUCCESS METRICS ---")
    print(f"Clusters with labeled MACs: {num_clusters_with_labels}")
    print(f'Pure clusters (single LABEL): {num_pure_clusters} ({pure_clusters_pct:.1f}%)')
    print(f"Devices recognized as a single device: {correctly_one_cluster_devices}/{total_devices} ({devices_one_cluster_pct:.1f}%)")

    # === DCMS: Device–Cluster Match Score (F1 from precision and recall) ===
    cluster_to_macs_labeled = {
        int(cid): set(mac for mac in macs if mac in mac_to_label)
        for cid, macs in clustered_data.items()
    }
    device_to_macs = {label: set(macs) for label, macs in label_to_macs.items()}

    def dcms_for_pair(cluster_macs, device_macs):
        inter = cluster_macs & device_macs
        if not inter:
            return 0.0, 0, 0, 0
        P = len(inter) / len(cluster_macs) if cluster_macs else 0.0
        R = len(inter) / len(device_macs) if device_macs else 0.0
        F1 = (2*P*R)/(P+R) if (P+R) > 0 else 0.0
        return F1, len(inter), len(cluster_macs), len(device_macs)

    dcms_per_cluster = {}
    perfect_cluster_matches = 0

    for cid, c_macs in cluster_to_macs_labeled.items():
        best = (0.0, None, 0, 0, 0)  # (F1, device_label, inter, |C|, |D|)
        for label, d_macs in device_to_macs.items():
            F1, inter, c_size, d_size = dcms_for_pair(c_macs, d_macs)
            if F1 > best[0]:
                best = (F1, label, inter, c_size, d_size)
        dcms_per_cluster[cid] = {
            "best_label": best[1],
            "dcms": best[0],
            "intersection": best[2],
            "cluster_size": best[3],
            "device_size": best[4],
        }
        if best[0] == 1.0 and best[3] == best[4] == best[2]:
            perfect_cluster_matches += 1

    valid_clusters = [cid for cid, s in dcms_per_cluster.items() if s["cluster_size"] > 0]
    macro_avg_dcms = sum(dcms_per_cluster[cid]["dcms"] for cid in valid_clusters) / len(valid_clusters) if valid_clusters else 0.0
    weighted_avg_dcms = (
        sum(dcms_per_cluster[cid]["dcms"] * dcms_per_cluster[cid]["cluster_size"] for cid in valid_clusters) /
        sum(dcms_per_cluster[cid]["cluster_size"] for cid in valid_clusters)
    ) if valid_clusters else 0.0

    print("\n--- DCMS (Device–Cluster Match Score) ---")
    print(f"Clusters with labeled MACs: {len(valid_clusters)}")
    print(f"Average DCMS (macro): {macro_avg_dcms:.3f}")
    print(f"Average DCMS (weighted by |C|): {weighted_avg_dcms:.3f}")
    print(f"Perfect matches (DCMS=1 and C==D): {perfect_cluster_matches}")

    worst = sorted(
        ((cid, s["dcms"], s["best_label"], s["intersection"], s["cluster_size"], s["device_size"])
        for cid, s in dcms_per_cluster.items() if s["cluster_size"] > 0),
        key=lambda x: x[1]
    )[:10]

    if worst:
        print("\nLowest DCMS (Top 10):")
        for cid, f1, lbl, inter, csz, dsz in worst:
            print(f"Cluster {cid}: DCMS={f1:.3f}, best_label={lbl}, |C∩D|={inter}, |C|={csz}, |D|={dsz}")

    # === ERRORS PER CLUSTER ===
    incorrect_per_cluster = defaultdict(int)
    for item in incorrect_macs:
        incorrect_per_cluster[item["cluster"]] += 1

    print("\n--- CLUSTER ERRORS ---")
    if incorrect_per_cluster:
        for cluster_id, count in sorted(incorrect_per_cluster.items()):
            print(f"Cluster {cluster_id}: {count} incorrectly assigned MACs")
    else:
        print("No incorrectly assigned MACs in clusters.")

    # === SAVING INCORRECT ASSIGNMENTS ===
    if incorrect_macs:
        for item in incorrect_macs:
            item["formatted"] = f"{item['MAC']} (label: {item['label']}, cluster: {item['cluster']}, expected: {item['expected']})"
        with open("outputs/incorrect_macs.json", "w") as f:
            json.dump(incorrect_macs, f, indent=2)
        print("Saved incorrectly classified MACs to file: outputs/incorrect_macs.json")


    device_stats = []
    for label, macs in label_to_macs.items():
        macs_in_clusters = [mac for mac in macs if mac in mac_to_cluster]
        clusters = set(mac_to_cluster[mac] for mac in macs_in_clusters)

        device_stats.append({
            "label": label,
            "mac_count": len(macs_in_clusters),
            "clusters": list(clusters)
        })

    device_stats.sort(key=lambda x: x["label"])

    table_data = []
    for entry in device_stats:
        clusters = sorted(entry['clusters'])
        clusters_str = ", ".join(str(c) for c in clusters) if clusters else "-"
        table_data.append([
            entry['label'],
            entry['mac_count'],
            len(clusters),
            clusters_str
        ])

    headers = [
        "Label",
        "MACs Detected",
        "Devices Detected",
        "Clusters" 
    ]

    print("\n--- DEVICES AND CLUSTERS ---")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))



def _entropy_from_counts(counts_dict, total):
    """Shannon entropy in nats; the log base doesn't matter, as it cancels out in the ratios."""
    return -sum((n/total) * math.log(n/total) for n in counts_dict.values() if n > 0)

def _conditional_entropy_C_given_K(C_counts, K_counts, CK_counts, N):
    """H(C|K) = sum_k p(k)*H(C|k)"""
    H = 0.0
    for k, nk in K_counts.items():
        H_Ck = 0.0
        for c in C_counts.keys():
            n_ck = CK_counts.get((k, c), 0)
            if n_ck:
                p = n_ck / nk
                H_Ck -= p * math.log(p)
        H += (nk / N) * H_Ck
    return H

def _conditional_entropy_K_given_C(C_counts, K_counts, CK_counts, N):
    """H(K|C) = sum_c p(c)*H(K|c)"""
    H = 0.0
    for c, nc in C_counts.items():
        H_Kc = 0.0
        for k in K_counts.keys():
            n_ck = CK_counts.get((k, c), 0)
            if n_ck:
                p = n_ck / nc
                H_Kc -= p * math.log(p)
        H += (nc / N) * H_Kc
    return H

def compute_v_measure(clustered_data,
                      mac_to_label,
                      include_unlabeled=False,
                      unlabeled_class="__UNK__",
                      beta=1.0):
    """
    Computes h, c, V_beta and VI (Variation of Information) based on:
      - clustered_data: {cluster_id(str): [mac,...]}
      - mac_to_label: {mac_normalized -> label}
    include_unlabeled=True treats unlabeled MACs as the '__UNK__' class.
    """
    # Build the list of (k, c) pairs for the evaluated elements
    pairs = []
    for cid_str, mac_list in clustered_data.items():
        k = int(cid_str)
        for mac in mac_list:
            if mac in mac_to_label:
                pairs.append((k, mac_to_label[mac]))
            elif include_unlabeled:
                pairs.append((k, unlabeled_class))

    N = len(pairs)
    if N == 0:
        return {
            "N": 0, "h": float("nan"), "c": float("nan"), "v": float("nan"),
            "H_C": 0.0, "H_K": 0.0, "H_C_given_K": 0.0, "H_K_given_C": 0.0, "VI": 0.0,
            "num_classes": 0, "num_clusters": 0, "include_unlabeled": include_unlabeled
        }

    K_counts = Counter(k for k, _ in pairs)
    C_counts = Counter(c for _, c in pairs)
    CK_counts = defaultdict(int)
    for k, c in pairs:
        CK_counts[(k, c)] += 1

    H_C = _entropy_from_counts(C_counts, N)
    H_K = _entropy_from_counts(K_counts, N)
    H_C_given_K = _conditional_entropy_C_given_K(C_counts, K_counts, CK_counts, N)
    H_K_given_C = _conditional_entropy_K_given_C(C_counts, K_counts, CK_counts, N)

    # homogeneity and completeness (with edge cases handled)
    h = 1.0 if H_C == 0 else 1.0 - (H_C_given_K / H_C)
    c = 1.0 if H_K == 0 else 1.0 - (H_K_given_C / H_K)

    v = 0.0 if (beta * h + c) == 0 else (1.0 + beta) * h * c / (beta * h + c)
    VI = H_C_given_K + H_K_given_C  # Variation of Information (optional: additional information-theoretic measure)

    return {
        "N": N, "h": h, "c": c, "v": v, "beta": beta,
        "H_C": H_C, "H_K": H_K, "H_C_given_K": H_C_given_K, "H_K_given_C": H_K_given_C, "VI": VI,
        "num_classes": len(C_counts), "num_clusters": len(K_counts), "include_unlabeled": include_unlabeled
    }

def print_v_measure_report_simple(clustered_data, mac_to_label, beta=1.0):
    res = compute_v_measure(
        clustered_data=clustered_data,
        mac_to_label=mac_to_label,
        include_unlabeled=False,   # classic variant: ignores MACs without a label
        beta=beta
    )
    print("\n--- V-MEASURE ---")
    print(f"N={res['N']}, #classes={res['num_classes']}, #clusters={res['num_clusters']}")
    print(f"h={res['h']:.3f}  c={res['c']:.3f}  V-measure (β={beta})={res['v']:.3f}")


# V-measure validation
# === validation against the ready-made scikit-learn implementation ===

def validation_python(clustered_data, mac_to_label):
    try:
        from sklearn.metrics import v_measure_score, homogeneity_score, completeness_score
        _sklearn_ok = True
    except Exception as e:
        _sklearn_ok = False
        _sklearn_err = e

    if _sklearn_ok:
        # Build y_true / y_pred only for labeled MACs (classic variant)
        y_true, y_pred = [], []
        for cid_str, mac_list in clustered_data.items():
            cid = int(cid_str)
            for mac in mac_list:
                # mac_n = normalize_mac(mac)
                mac_n = mac
                if mac_n in mac_to_label:      # skip unlabeled
                    y_true.append(mac_to_label[mac_n])
                    y_pred.append(cid)

        if len(y_true) == 0:
            print("\n--- V-MEASURE (scikit-learn) ---")
            print("No labeled elements to compute on.")
        else:
            # scikit-learn result
            sk_h = homogeneity_score(y_true, y_pred)
            sk_c = completeness_score(y_true, y_pred)
            sk_v = v_measure_score(y_true, y_pred, beta=1.0)

            # Result of our implementation (classic variant without UNK) - for comparison
            ours = compute_v_measure(
                clustered_data=clustered_data,
                mac_to_label=mac_to_label,
                include_unlabeled=False,
                beta=1.0
            )

            print("\n--- V-MEASURE (scikit-learn) ---")
            print(f"N={len(y_true)}, #classes={len(set(y_true))}, #clusters={len(set(y_pred))}")
            print(f"h={sk_h:.3f}  c={sk_c:.3f}  V-measure (β=1.0)={sk_v:.3f}")

            # Compare values (scikit-learn vs ours)
            dh = sk_h - ours['h']
            dc = sk_c - ours['c']
            dv = sk_v - ours['v']
            print("\nComparison with our implementation (sklearn - ours):")
            print(f"Δh={dh:.6f}  Δc={dc:.6f}  ΔV={dv:.6f}")

    else:
        print("\n--- V-MEASURE (scikit-learn) ---")
        print("scikit-learn is missing. Install it with: pip install scikit-learn")
        print(f"Import error details: {_sklearn_err}")


def display_reverted_table(clustered_data, mac_to_label):
    """
    Display a table showing clusters and the labels/devices assigned to each cluster.
    
    Parameters:
    - clustered_data: {cluster_id(str): [mac,...]}
    - mac_to_label: {mac -> label}
    """
    
    # Build cluster_to_labels mapping
    cluster_to_labels = defaultdict(set)
    cluster_mac_counts = {}
    for cluster_id_str, mac_list in clustered_data.items():
        cluster_id = int(cluster_id_str)
        cluster_mac_counts[cluster_id] = len(mac_list)
        for mac in mac_list:
            if mac in mac_to_label:
                cluster_to_labels[cluster_id].add(mac_to_label[mac])
    
    # Sort clusters by ID
    sorted_clusters = sorted(cluster_mac_counts.keys())
    
    # Build table data
    table_data = []
    for cluster_id in sorted_clusters:
        labels = sorted(cluster_to_labels.get(cluster_id, []))
        labels_str = ", ".join(labels) if labels else "-"
        table_data.append([
            cluster_id,
            cluster_mac_counts.get(cluster_id, 0),
            labels_str
        ])
    
    headers = ["Cluster", "MAC Count", "Assigned Labels"]
    
    print("\n--- CLUSTERS AND ASSIGNED DEVICES ---")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

