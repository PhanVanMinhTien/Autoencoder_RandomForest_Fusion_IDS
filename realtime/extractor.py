# from nfstream import NFStreamer
# import pandas as pd
# import os

# INTERFACE = "s1-eth10" 
# OUTPUT_FILE = "collected_data.csv"
# BUFFER_SIZE = 1 
# flow_buffer = []

# def extract_features():
#     print(f"[*] Đang trích xuất đặc trưng trên {INTERFACE}...")
#     streamer = NFStreamer(source=INTERFACE, statistical_analysis=True, splt_analysis=0, idle_timeout=5)

#     for flow in streamer:

#         # Only keep TCP (6) and UDP (17) flows
#         if flow.protocol not in [6, 17]:
#             continue
#         # Skip IPv6 multicast/broadcast addresses
#         if str(flow.dst_ip).startswith("ff"):
#             continue
#         # Skip flows with no bidirectional packets (e.g., unidirectional or incomplete flows)
#         if flow.bidirectional_packets < 1:
#             continue

#         d = flow.bidirectional_duration_ms / 1000.0
#         if d <= 0:
#             d = 0.0001
#         # MAPPING 75 ĐẶC TRƯNG
#         # data = {
#         #     "Flow ID": f"{flow.src_ip}-{flow.dst_ip}-{flow.src_port}-{flow.dst_port}-{flow.protocol}",
#         #     "Source IP": flow.src_ip, "Source Port": flow.src_port,
#         #     "Destination IP": flow.dst_ip, "Destination Port": flow.dst_port,
#         #     "Protocol": flow.protocol, "Timestamp": flow.bidirectional_first_seen_ms,
#         #     "Flow Duration": flow.bidirectional_duration_ms,
#         #     "Total Fwd Packets": flow.src2dst_packets,
#         #     "Total Backward Packets": flow.dst2src_packets,
#         #     "Total Length of Fwd Packets": flow.src2dst_bytes,
#         #     "Total Length of Bwd Packets": flow.dst2src_bytes,
#         #     "Fwd Packet Length Max": flow.src2dst_max_ps,
#         #     "Fwd Packet Length Min": flow.src2dst_min_ps,
#         #     "Fwd Packet Length Mean": flow.src2dst_mean_ps,
#         #     "Fwd Packet Length Std": flow.src2dst_stddev_ps,
#         #     "Bwd Packet Length Max": flow.dst2src_max_ps,
#         #     "Bwd Packet Length Min": flow.dst2src_min_ps,
#         #     "Bwd Packet Length Mean": flow.dst2src_mean_ps,
#         #     "Bwd Packet Length Std": flow.dst2src_stddev_ps,
#         #     "Flow Bytes/s": flow.bidirectional_bytes / (d + 0.0001),
#         #     "Flow Packets/s": flow.bidirectional_packets / (d + 0.0001),
#         #     "Flow IAT Mean": flow.bidirectional_mean_piat_ms,
#         #     "Flow IAT Std": flow.bidirectional_stddev_piat_ms,
#         #     "Flow IAT Max": flow.bidirectional_max_piat_ms,
#         #     "Flow IAT Min": flow.bidirectional_min_piat_ms,
#         #     "Fwd IAT Total": flow.src2dst_duration_ms,
#         #     "Fwd IAT Mean": flow.src2dst_mean_piat_ms,
#         #     "Fwd IAT Std": flow.src2dst_stddev_piat_ms,
#         #     "Fwd IAT Max": flow.src2dst_max_piat_ms,
#         #     "Fwd IAT Min": flow.src2dst_min_piat_ms,
#         #     "Bwd IAT Total": flow.dst2src_duration_ms,
#         #     "Bwd IAT Mean": flow.dst2src_mean_piat_ms,
#         #     "Bwd IAT Std": flow.dst2src_stddev_piat_ms,
#         #     "Bwd IAT Max": flow.dst2src_max_piat_ms,
#         #     "Bwd IAT Min": flow.dst2src_min_piat_ms,
#         #     "Fwd PSH Flags": flow.src2dst_psh_packets,
#         #     "Bwd PSH Flags": flow.dst2src_psh_packets,
#         #     "Fwd URG Flags": flow.src2dst_urg_packets,
#         #     "Bwd URG Flags": flow.dst2src_urg_packets,
#         #     # --- ĐÃ SỬA: Dùng _size thay vì _bytes ---
#         #     "Fwd Header Length": getattr(flow, 'src2dst_header_size', 0),
#         #     "Bwd Header Length": getattr(flow, 'dst2src_header_size', 0),
#         #     "Packet Length Min": flow.bidirectional_min_ps,
#         #     "Packet Length Max": flow.bidirectional_max_ps,
#         #     "Packet Length Mean": flow.bidirectional_mean_ps,
#         #     "Packet Length Std": flow.bidirectional_stddev_ps,
#         #     "Packet Length Variance": flow.bidirectional_stddev_ps**2,
#         #     "Average Packet Size": flow.bidirectional_bytes / (flow.bidirectional_packets + 0.001),
#         #     "Avg Fwd Segment Size": flow.src2dst_mean_ps,
#         #     "Avg Bwd Segment Size": flow.dst2src_mean_ps,
#         #     "Init_Win_bytes_forward": getattr(flow, 'src2dst_syn_packets', 0), 
#         #     "Init_Win_bytes_backward": getattr(flow, 'dst2src_syn_packets', 0),
#         #     "act_data_pkt_fwd": flow.src2dst_packets,
#         #     "min_seg_size_forward": flow.src2dst_min_ps,  
#         # }
#         # data["PSH Flag Count"] = flow.src2dst_psh_packets + flow.dst2src_psh_packets
#         # data["Bwd Packets/s"] = flow.dst2src_packets / (d + 1e-6)
#         # data["Down/Up Ratio"] = flow.dst2src_packets / (flow.src2dst_packets + 1)
#         # data["Min Packet Length"] = flow.bidirectional_min_ps
#         # # Approx
#         # data["RST Flag Count"] = 0
#         # data["SYN Flag Count"] = getattr(flow, 'src2dst_syn_packets', 0)
#         # data["ECE Flag Count"] = 0
#         # data["URG Flag Count"] = flow.src2dst_urg_packets + flow.dst2src_urg_packets
#         # # Không có → fill 0
#         # data["Idle Std"] = 0
#         # data["Active Min"] = 0
#         # data["Active Mean"] = 0

#         data = {
#         # ===== Statistical features =====
#         "Total Length of Fwd Packets": flow.src2dst_bytes,
#         "Bwd IAT Min": flow.dst2src_min_piat_ms,
#         "act_data_pkt_fwd": flow.src2dst_packets,
#         "Bwd Packet Length Min": flow.dst2src_min_ps,
#         "Total Fwd Packets": flow.src2dst_packets,
#         "Bwd IAT Mean": flow.dst2src_mean_piat_ms,
#         "Destination Port": flow.dst_port,
#         "Flow IAT Std": flow.bidirectional_stddev_piat_ms,
#         "Bwd Packet Length Std": flow.dst2src_stddev_ps,
#         "Bwd IAT Max": flow.dst2src_max_piat_ms,
#         "Fwd Packet Length Max": flow.src2dst_max_ps,
#         "Fwd PSH Flags": flow.src2dst_psh_packets,
#         "Init_Win_bytes_backward": getattr(flow, 'dst2src_syn_packets', 0),
#         "Flow Duration": flow.bidirectional_duration_ms,
#         "Fwd IAT Min": flow.src2dst_min_piat_ms,
#         "Bwd IAT Std": flow.dst2src_stddev_piat_ms,
#         "Fwd Packet Length Std": flow.src2dst_stddev_ps,
#         "Fwd IAT Total": flow.src2dst_duration_ms,
#         "Fwd IAT Mean": flow.src2dst_mean_piat_ms,
#         "Min Packet Length": flow.bidirectional_min_ps,

#         # ===== Derived features =====
#         "PSH Flag Count": flow.src2dst_psh_packets + flow.dst2src_psh_packets,
#         "Bwd Packets/s": flow.dst2src_packets / (d + 1e-6),
#         "Down/Up Ratio": flow.dst2src_packets / (flow.src2dst_packets + 1),

#         # ===== Flag features (approx / partial) =====
#         "SYN Flag Count": getattr(flow, 'src2dst_syn_packets', 0),
#         "URG Flag Count": flow.src2dst_urg_packets + flow.dst2src_urg_packets,

#         # ===== Missing features → fill 0 =====
#         "RST Flag Count": 0,
#         "ECE Flag Count": 0,
#         "Idle Std": 0,
#         "Active Min": 0,
#         "Active Mean": 0,
#         }


#         flow_buffer.append(data)
#         if len(flow_buffer) >= BUFFER_SIZE:
#             df = pd.DataFrame(flow_buffer)
#             df.to_csv(OUTPUT_FILE, mode='a', header=not os.path.exists(OUTPUT_FILE), index=False)
#             flow_buffer.clear()
#             print(f"[OK] Đã lưu {BUFFER_SIZE} luồng vào CSV.", flush=True)

# if __name__ == "__main__":
#     try:
#         extract_features()
#     except KeyboardInterrupt:
#         if flow_buffer:
#             pd.DataFrame(flow_buffer).to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
#         print("\n[!] Dừng trích xuất.")


# extractor.py

from nfstream import NFStreamer
import pandas as pd
import os
import argparse
import time


FEATURE_30 = [
    "RST Flag Count",
    "Total Length of Fwd Packets",
    "Bwd IAT Min",
    "ECE Flag Count",
    "act_data_pkt_fwd",
    "Idle Std",
    "Bwd Packet Length Min",
    "Total Fwd Packets",
    "Bwd IAT Mean",
    "PSH Flag Count",
    "Destination Port",
    "Flow IAT Std",
    "Bwd Packet Length Std",
    "Bwd IAT Max",
    "Fwd Packet Length Max",
    "Fwd PSH Flags",
    "Active Min",
    "Init_Win_bytes_backward",
    "SYN Flag Count",
    "Flow Duration",
    "Fwd IAT Min",
    "Down/Up Ratio",
    "Bwd IAT Std",
    "Fwd Packet Length Std",
    "Fwd IAT Total",
    "Bwd Packets/s",
    "Active Mean",
    "Fwd IAT Mean",
    "URG Flag Count",
    "Min Packet Length",
]


def g(flow, attr, default=0):
    """
    Safe getattr for NFStreamer attributes.
    Some attributes may not exist depending on NFStreamer version.
    """
    return getattr(flow, attr, default)


def build_flow_record(flow):
    duration_sec = flow.bidirectional_duration_ms / 1000.0
    if duration_sec <= 0:
        duration_sec = 0.0001

    src2dst_packets = g(flow, "src2dst_packets", 0)
    dst2src_packets = g(flow, "dst2src_packets", 0)

    record = {
        # ===== Metadata =====
        "Flow ID": f"{flow.src_ip}-{flow.dst_ip}-{flow.src_port}-{flow.dst_port}-{flow.protocol}",
        "Source IP": flow.src_ip,
        "Source Port": flow.src_port,
        "Destination IP": flow.dst_ip,
        "Destination Port": flow.dst_port,
        "Protocol": flow.protocol,
        "Timestamp": g(flow, "bidirectional_first_seen_ms", int(time.time() * 1000)),

        # ===== 30 selected features =====
        "RST Flag Count": g(flow, "src2dst_rst_packets", 0) + g(flow, "dst2src_rst_packets", 0),
        "Total Length of Fwd Packets": g(flow, "src2dst_bytes", 0),
        "Bwd IAT Min": g(flow, "dst2src_min_piat_ms", 0),
        "ECE Flag Count": g(flow, "src2dst_ece_packets", 0) + g(flow, "dst2src_ece_packets", 0),
        "act_data_pkt_fwd": src2dst_packets,
        "Idle Std": 0,
        "Bwd Packet Length Min": g(flow, "dst2src_min_ps", 0),
        "Total Fwd Packets": src2dst_packets,
        "Bwd IAT Mean": g(flow, "dst2src_mean_piat_ms", 0),
        "PSH Flag Count": g(flow, "src2dst_psh_packets", 0) + g(flow, "dst2src_psh_packets", 0),
        "Destination Port": flow.dst_port,
        "Flow IAT Std": g(flow, "bidirectional_stddev_piat_ms", 0),
        "Bwd Packet Length Std": g(flow, "dst2src_stddev_ps", 0),
        "Bwd IAT Max": g(flow, "dst2src_max_piat_ms", 0),
        "Fwd Packet Length Max": g(flow, "src2dst_max_ps", 0),
        "Fwd PSH Flags": g(flow, "src2dst_psh_packets", 0),
        "Active Min": 0,
        "Init_Win_bytes_backward": 0,
        "SYN Flag Count": g(flow, "src2dst_syn_packets", 0) + g(flow, "dst2src_syn_packets", 0),
        "Flow Duration": g(flow, "bidirectional_duration_ms", 0),
        "Fwd IAT Min": g(flow, "src2dst_min_piat_ms", 0),
        "Down/Up Ratio": dst2src_packets / (src2dst_packets + 1),
        "Bwd IAT Std": g(flow, "dst2src_stddev_piat_ms", 0),
        "Fwd Packet Length Std": g(flow, "src2dst_stddev_ps", 0),
        "Fwd IAT Total": g(flow, "src2dst_duration_ms", 0),
        "Bwd Packets/s": dst2src_packets / (duration_sec + 1e-6),
        "Active Mean": 0,
        "Fwd IAT Mean": g(flow, "src2dst_mean_piat_ms", 0),
        "URG Flag Count": g(flow, "src2dst_urg_packets", 0) + g(flow, "dst2src_urg_packets", 0),
        "Min Packet Length": g(flow, "bidirectional_min_ps", 0),
    }

    return record


def append_records(records, output_file):
    if not records:
        return

    df = pd.DataFrame(records)
    file_exists = os.path.exists(output_file)

    df.to_csv(
        output_file,
        mode="a",
        header=not file_exists,
        index=False
    )


def extract_features(interface, output_file, buffer_size, idle_timeout):
    print(f"[*] Monitoring interface: {interface}")
    print(f"[*] Output file: {output_file}")
    print("[*] Press Ctrl+C to stop.")

    flow_buffer = []

    streamer = NFStreamer(
        source=interface,
        statistical_analysis=True,
        splt_analysis=0,
        idle_timeout=idle_timeout
    )

    for flow in streamer:
        # Keep TCP/UDP only
        if flow.protocol not in [6, 17]:
            continue

        # Skip IPv6 multicast/broadcast
        if str(flow.dst_ip).startswith("ff"):
            continue

        # Skip invalid/empty flows
        if g(flow, "bidirectional_packets", 0) < 1:
            continue

        record = build_flow_record(flow)
        flow_buffer.append(record)

        if len(flow_buffer) >= buffer_size:
            append_records(flow_buffer, output_file)
            print(
                f"[OK] Saved {len(flow_buffer)} flow(s). "
                f"Last flow: {record['Source IP']} -> {record['Destination IP']}",
                flush=True
            )
            flow_buffer.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface", default="s1-eth10")
    parser.add_argument("--output", default="collected_data.csv")
    parser.add_argument("--buffer-size", type=int, default=1)
    parser.add_argument("--idle-timeout", type=int, default=5)

    args = parser.parse_args()

    try:
        extract_features(
            interface=args.interface,
            output_file=args.output,
            buffer_size=args.buffer_size,
            idle_timeout=args.idle_timeout
        )
    except KeyboardInterrupt:
        print("\n[!] Extractor stopped.")