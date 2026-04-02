# from nfstream import NFStreamer
# import pandas as pd
# import os
# import time

# # Cấu hình
# INTERFACE = "s1-eth1" 
# OUTPUT_FILE = "cic_ids_2017_full.csv"

# def extract_features():
#     print(f"[*] Đang bắt đầu trích xuất ~75 đặc trưng trên {INTERFACE}...")
    
#     # Khởi tạo streamer với đầy đủ cấu hình thống kê
#     # streamer = NFStreamer(source=INTERFACE, 
#     #                       statistical_analysis=True,
#     #                       splt_analysis=10,
#     #                       idle_timeout=10,
#     #                       active_timeout=30)
    
#     streamer = NFStreamer(source=INTERFACE, 
#                       n_proc=4,  # Sử dụng 4 luồng xử lý song song
#                       statistical_analysis=True,
#                       idle_timeout=1,   # Giảm tối đa để xuất flow ngay khi ngừng gói tin
#                       active_timeout=5) # Cắt nhỏ các luồng dài để xử lý sớm hơn

#     for flow in streamer:
#         # Chuyển đổi mili giây sang giây cho các phép tính tốc độ
#         duration_s = flow.bidirectional_duration_ms / 1000.0
        
#         # Dictionary ánh xạ chính xác thuộc tính NFStream -> CIC-IDS
#         row = {
#             # --- Nhóm nhận dạng (Metadata) ---
#             "Flow ID": f"{flow.src_ip}-{flow.dst_ip}-{flow.src_port}-{flow.dst_port}-{flow.protocol}",
#             "Source IP": flow.src_ip,
#             "Source Port": flow.src_port,
#             "Destination IP": flow.dst_ip,
#             "Destination Port": flow.dst_port,
#             "Protocol": flow.protocol,
#             "Timestamp": flow.bidirectional_first_seen_ms, # ĐÃ SỬA LỖI Ở ĐÂY
            
#             # --- Nhóm thời gian (Duration & IAT) ---
#             "Flow Duration": flow.bidirectional_duration_ms,
#             "Flow IAT Mean": flow.bidirectional_mean_piat,
#             "Flow IAT Std": flow.bidirectional_stddev_piat,
#             "Flow IAT Max": flow.bidirectional_max_piat,
#             "Flow IAT Min": flow.bidirectional_min_piat,
#             "Fwd IAT Total": flow.src2dst_duration_ms,
#             "Fwd IAT Mean": flow.src2dst_mean_piat,
#             "Fwd IAT Std": flow.src2dst_stddev_piat,
#             "Fwd IAT Max": flow.src2dst_max_piat,
#             "Fwd IAT Min": flow.src2dst_min_piat,
#             "Bwd IAT Total": flow.dst2src_duration_ms,
#             "Bwd IAT Mean": flow.dst2src_mean_piat,
#             "Bwd IAT Std": flow.dst2src_stddev_piat,
#             "Bwd IAT Max": flow.dst2src_max_piat,
#             "Bwd IAT Min": flow.dst2src_min_piat,
            
#             # --- Nhóm số lượng gói tin và kích thước ---
#             "Total Fwd Packets": flow.src2dst_packets,
#             "Total Backward Packets": flow.dst2src_packets,
#             "Total Length of Fwd Packets": flow.src2dst_bytes,
#             "Total Length of Bwd Packets": flow.dst2src_bytes,
#             "Fwd Packet Length Max": flow.src2dst_max_ps,
#             "Fwd Packet Length Min": flow.src2dst_min_ps,
#             "Fwd Packet Length Mean": flow.src2dst_mean_ps,
#             "Fwd Packet Length Std": flow.src2dst_stddev_ps,
#             "Bwd Packet Length Max": flow.dst2src_max_ps,
#             "Bwd Packet Length Min": flow.dst2src_min_ps,
#             "Bwd Packet Length Mean": flow.dst2src_mean_ps,
#             "Bwd Packet Length Std": flow.dst2src_stddev_ps,
            
#             # --- Nhóm tốc độ (Rates) ---
#             "Flow Bytes/s": flow.bidirectional_bytes / (duration_s + 0.0001),
#             "Flow Packets/s": flow.bidirectional_packets / (duration_s + 0.0001),
            
#             # --- Nhóm Flags & Header ---
#             "Fwd PSH Flags": flow.src2dst_psh_packets,
#             "Bwd PSH Flags": flow.dst2src_psh_packets,
#             "Fwd URG Flags": flow.src2dst_urg_packets,
#             "Bwd URG Flags": flow.dst2src_urg_packets,
#             "Fwd Header Length": flow.src2dst_header_bytes,
#             "Bwd Header Length": flow.dst2src_header_bytes,
            
#             # --- Đặc trưng thống kê bổ sung ---
#             "Packet Length Min": flow.bidirectional_min_ps,
#             "Packet Length Max": flow.bidirectional_max_ps,
#             "Packet Length Mean": flow.bidirectional_mean_ps,
#             "Packet Length Std": flow.bidirectional_stddev_ps,
#             "Packet Length Variance": flow.bidirectional_stddev_ps**2,
#             "Average Packet Size": flow.bidirectional_bytes / (flow.bidirectional_packets + 0.001),
#             "Avg Fwd Segment Size": flow.src2dst_mean_ps,
#             "Avg Bwd Segment Size": flow.dst2src_mean_ps,
#             "Init_Win_bytes_forward": flow.src2dst_syn_packets, 
#             "Init_Win_bytes_backward": flow.dst2src_syn_packets,
#             "act_data_pkt_fwd": flow.src2dst_packets,
#             "min_seg_size_forward": flow.src2dst_min_ps,
#         }
        
#         # Xuất kết quả
#         df = pd.DataFrame([row])
#         file_exists = os.path.isfile(OUTPUT_FILE)
#         df.to_csv(OUTPUT_FILE, mode='a', header=not file_exists, index=False)
#         print(f"[OK] Đã ghi luồng: {row['Source IP']} -> {row['Destination IP']} ({row['Protocol']})", flush=True)

# if __name__ == "__main__":
#     try:
#         extract_features()
#     except KeyboardInterrupt:
#         print("\n[!] Đã dừng chương trình.")



# from nfstream import NFStreamer
# import pandas as pd
# import os
# import time

# INTERFACE = "s1-eth1" 
# OUTPUT_FILE = "cic_ids_2017_full.csv"
# BUFFER_SIZE = 10  # Ghi file sau mỗi 10 luồng để tăng tốc
# flow_buffer = []

# def extract_features():
#     print(f"[*] Đang chạy trích xuất đặc trưng trên {INTERFACE}...")
    
#     # Loại bỏ n_proc/n_dissectors để tránh lỗi TypeError
#     streamer = NFStreamer(source=INTERFACE, 
#                           statistical_analysis=True,
#                           splt_analysis=0, # Giảm xuống 0 nếu không cần phân tích gói tin đầu để tăng tốc
#                           idle_timeout=1)

#     for flow in streamer:
#         duration_s = flow.bidirectional_duration_ms / 1000.0
        
#         # Trích xuất 75 đặc trưng (rút gọn ví dụ, bạn giữ nguyên list 75 cột cũ)
#         data = {
#             # --- Nhóm nhận dạng (Metadata) ---
#             "Flow ID": f"{flow.src_ip}-{flow.dst_ip}-{flow.src_port}-{flow.dst_port}-{flow.protocol}",
#             "Source IP": flow.src_ip,
#             "Source Port": flow.src_port,
#             "Destination IP": flow.dst_ip,
#             "Destination Port": flow.dst_port,
#             "Protocol": flow.protocol,
#             "Timestamp": flow.bidirectional_first_seen_ms, # ĐÃ SỬA LỖI Ở ĐÂY
            
#             # --- Nhóm thời gian (Duration & IAT) ---
#             "Flow Duration": flow.bidirectional_duration_ms,
#             "Flow IAT Mean": flow.bidirectional_mean_piat,
#             "Flow IAT Std": flow.bidirectional_stddev_piat,
#             "Flow IAT Max": flow.bidirectional_max_piat,
#             "Flow IAT Min": flow.bidirectional_min_piat,
#             "Fwd IAT Total": flow.src2dst_duration_ms,
#             "Fwd IAT Mean": flow.src2dst_mean_piat,
#             "Fwd IAT Std": flow.src2dst_stddev_piat,
#             "Fwd IAT Max": flow.src2dst_max_piat,
#             "Fwd IAT Min": flow.src2dst_min_piat,
#             "Bwd IAT Total": flow.dst2src_duration_ms,
#             "Bwd IAT Mean": flow.dst2src_mean_piat,
#             "Bwd IAT Std": flow.dst2src_stddev_piat,
#             "Bwd IAT Max": flow.dst2src_max_piat,
#             "Bwd IAT Min": flow.dst2src_min_piat,
            
#             # --- Nhóm số lượng gói tin và kích thước ---
#             "Total Fwd Packets": flow.src2dst_packets,
#             "Total Backward Packets": flow.dst2src_packets,
#             "Total Length of Fwd Packets": flow.src2dst_bytes,
#             "Total Length of Bwd Packets": flow.dst2src_bytes,
#             "Fwd Packet Length Max": flow.src2dst_max_ps,
#             "Fwd Packet Length Min": flow.src2dst_min_ps,
#             "Fwd Packet Length Mean": flow.src2dst_mean_ps,
#             "Fwd Packet Length Std": flow.src2dst_stddev_ps,
#             "Bwd Packet Length Max": flow.dst2src_max_ps,
#             "Bwd Packet Length Min": flow.dst2src_min_ps,
#             "Bwd Packet Length Mean": flow.dst2src_mean_ps,
#             "Bwd Packet Length Std": flow.dst2src_stddev_ps,
            
#             # --- Nhóm tốc độ (Rates) ---
#             "Flow Bytes/s": flow.bidirectional_bytes / (duration_s + 0.0001),
#             "Flow Packets/s": flow.bidirectional_packets / (duration_s + 0.0001),
            
#             # --- Nhóm Flags & Header ---
#             "Fwd PSH Flags": flow.src2dst_psh_packets,
#             "Bwd PSH Flags": flow.dst2src_psh_packets,
#             "Fwd URG Flags": flow.src2dst_urg_packets,
#             "Bwd URG Flags": flow.dst2src_urg_packets,
#             "Fwd Header Length": flow.src2dst_header_bytes,
#             "Bwd Header Length": flow.dst2src_header_bytes,
            
#             # --- Đặc trưng thống kê bổ sung ---
#             "Packet Length Min": flow.bidirectional_min_ps,
#             "Packet Length Max": flow.bidirectional_max_ps,
#             "Packet Length Mean": flow.bidirectional_mean_ps,
#             "Packet Length Std": flow.bidirectional_stddev_ps,
#             "Packet Length Variance": flow.bidirectional_stddev_ps**2,
#             "Average Packet Size": flow.bidirectional_bytes / (flow.bidirectional_packets + 0.001),
#             "Avg Fwd Segment Size": flow.src2dst_mean_ps,
#             "Avg Bwd Segment Size": flow.dst2src_mean_ps,
#             "Init_Win_bytes_forward": flow.src2dst_syn_packets, 
#             "Init_Win_bytes_backward": flow.dst2src_syn_packets,
#             "act_data_pkt_fwd": flow.src2dst_packets,
#             "min_seg_size_forward": flow.src2dst_min_ps,
#         }
        
#         flow_buffer.append(data)
#         print(f"[+] Captured: {len(flow_buffer)}/{BUFFER_SIZE}", end='\r')

#         if len(flow_buffer) >= BUFFER_SIZE:
#             df = pd.DataFrame(flow_buffer)
#             df.to_csv(OUTPUT_FILE, mode='a', header=not os.path.exists(OUTPUT_FILE), index=False)
#             flow_buffer.clear()
#             print(f"\n[OK] Đã lưu {BUFFER_SIZE} luồng vào CSV.")

# if __name__ == "__main__":
#     try:
#         extract_features()
#     except KeyboardInterrupt:
#         if flow_buffer:
#             pd.DataFrame(flow_buffer).to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
#         print("\n[!] Dừng và lưu dữ daily.")

from nfstream import NFStreamer
import pandas as pd
import os

INTERFACE = "s4-eth4" 
OUTPUT_FILE = "collected_data.csv"
BUFFER_SIZE = 10 
flow_buffer = []

def extract_features():
    print(f"[*] Đang trích xuất ~75 đặc trưng trên {INTERFACE}...")
    streamer = NFStreamer(source=INTERFACE, statistical_analysis=True, splt_analysis=0, idle_timeout=1)

    for flow in streamer:

        # # Only keep TCP (6) and UDP (17) flows
        # if flow.protocol not in [6, 17]:
        #     continue
        # # Skip IPv6 multicast/broadcast addresses
        # if str(flow.dst_ip).startswith("ff"):
        #     continue
        # # Skip flows with no bidirectional packets (e.g., unidirectional or incomplete flows)
        # if flow.bidirectional_packets == 0:
        #     continue

        d = flow.bidirectional_duration_ms / 1000.0
        if d <= 0:
            d = 0.0001
        # MAPPING 75 ĐẶC TRƯNG - BẢN 6.6.0 (Sửa lỗi _size và _ms)
        data = {
            "Flow ID": f"{flow.src_ip}-{flow.dst_ip}-{flow.src_port}-{flow.dst_port}-{flow.protocol}",
            "Source IP": flow.src_ip, "Source Port": flow.src_port,
            "Destination IP": flow.dst_ip, "Destination Port": flow.dst_port,
            "Protocol": flow.protocol, "Timestamp": flow.bidirectional_first_seen_ms,
            "Flow Duration": flow.bidirectional_duration_ms,
            "Total Fwd Packets": flow.src2dst_packets,
            "Total Backward Packets": flow.dst2src_packets,
            "Total Length of Fwd Packets": flow.src2dst_bytes,
            "Total Length of Bwd Packets": flow.dst2src_bytes,
            "Fwd Packet Length Max": flow.src2dst_max_ps,
            "Fwd Packet Length Min": flow.src2dst_min_ps,
            "Fwd Packet Length Mean": flow.src2dst_mean_ps,
            "Fwd Packet Length Std": flow.src2dst_stddev_ps,
            "Bwd Packet Length Max": flow.dst2src_max_ps,
            "Bwd Packet Length Min": flow.dst2src_min_ps,
            "Bwd Packet Length Mean": flow.dst2src_mean_ps,
            "Bwd Packet Length Std": flow.dst2src_stddev_ps,
            "Flow Bytes/s": flow.bidirectional_bytes / (d + 0.0001),
            "Flow Packets/s": flow.bidirectional_packets / (d + 0.0001),
            "Flow IAT Mean": flow.bidirectional_mean_piat_ms,
            "Flow IAT Std": flow.bidirectional_stddev_piat_ms,
            "Flow IAT Max": flow.bidirectional_max_piat_ms,
            "Flow IAT Min": flow.bidirectional_min_piat_ms,
            "Fwd IAT Total": flow.src2dst_duration_ms,
            "Fwd IAT Mean": flow.src2dst_mean_piat_ms,
            "Fwd IAT Std": flow.src2dst_stddev_piat_ms,
            "Fwd IAT Max": flow.src2dst_max_piat_ms,
            "Fwd IAT Min": flow.src2dst_min_piat_ms,
            "Bwd IAT Total": flow.dst2src_duration_ms,
            "Bwd IAT Mean": flow.dst2src_mean_piat_ms,
            "Bwd IAT Std": flow.dst2src_stddev_piat_ms,
            "Bwd IAT Max": flow.dst2src_max_piat_ms,
            "Bwd IAT Min": flow.dst2src_min_piat_ms,
            "Fwd PSH Flags": flow.src2dst_psh_packets,
            "Bwd PSH Flags": flow.dst2src_psh_packets,
            "Fwd URG Flags": flow.src2dst_urg_packets,
            "Bwd URG Flags": flow.dst2src_urg_packets,
            # --- ĐÃ SỬA: Dùng _size thay vì _bytes ---
            "Fwd Header Length": getattr(flow, 'src2dst_header_size', 0),
            "Bwd Header Length": getattr(flow, 'dst2src_header_size', 0),
            "Packet Length Min": flow.bidirectional_min_ps,
            "Packet Length Max": flow.bidirectional_max_ps,
            "Packet Length Mean": flow.bidirectional_mean_ps,
            "Packet Length Std": flow.bidirectional_stddev_ps,
            "Packet Length Variance": flow.bidirectional_stddev_ps**2,
            "Average Packet Size": flow.bidirectional_bytes / (flow.bidirectional_packets + 0.001),
            "Avg Fwd Segment Size": flow.src2dst_mean_ps,
            "Avg Bwd Segment Size": flow.dst2src_mean_ps,
            "Init_Win_bytes_forward": getattr(flow, 'src2dst_syn_packets', 0), 
            "Init_Win_bytes_backward": getattr(flow, 'dst2src_syn_packets', 0),
            "act_data_pkt_fwd": flow.src2dst_packets,
            "min_seg_size_forward": flow.src2dst_min_ps,
        }
        
        flow_buffer.append(data)
        if len(flow_buffer) >= BUFFER_SIZE:
            df = pd.DataFrame(flow_buffer)
            df.to_csv(OUTPUT_FILE, mode='a', header=not os.path.exists(OUTPUT_FILE), index=False)
            flow_buffer.clear()
            print(f"[OK] Đã lưu {BUFFER_SIZE} luồng vào CSV.", flush=True)

if __name__ == "__main__":
    try:
        extract_features()
    except KeyboardInterrupt:
        if flow_buffer:
            pd.DataFrame(flow_buffer).to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
        print("\n[!] Dừng trích xuất.")