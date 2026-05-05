# import torch
# import joblib
# import numpy as np
# import pandas as pd
# from nfstream import NFStreamer
# from ryu.base import app_manager
# from ryu.controller import ofp_event
# from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
# from ryu.controller.handler import set_ev_cls
# from ryu.ofproto import ofproto_v1_3
# from ryu.lib import hub
# from ryu.lib.packet import packet, ethernet, ipv4

# # Import kiến trúc Autoencoder từ Giai đoạn 1
# import torch.nn as nn

# class DeepAutoencoder(nn.Module):
#     def __init__(self, input_dim, latent_dim, hidden_layers):
#         super(DeepAutoencoder, self).__init__()
#         encoder_modules = []
#         current_dim = input_dim
#         for h_dim in hidden_layers:
#             encoder_modules.append(nn.Linear(current_dim, h_dim))
#             encoder_modules.append(nn.BatchNorm1d(h_dim))
#             encoder_modules.append(nn.LeakyReLU(0.2))
#             current_dim = h_dim
#         encoder_modules.append(nn.Linear(current_dim, latent_dim))
#         self.encoder = nn.Sequential(*encoder_modules)

#         decoder_modules = []
#         current_dim = latent_dim
#         reversed_hidden = hidden_layers[::-1]
#         for h_dim in reversed_hidden:
#             decoder_modules.append(nn.Linear(current_dim, h_dim))
#             decoder_modules.append(nn.BatchNorm1d(h_dim))
#             decoder_modules.append(nn.LeakyReLU(0.2))
#             current_dim = h_dim
#         decoder_modules.append(nn.Linear(current_dim, input_dim))
#         self.decoder = nn.Sequential(*decoder_modules)

#     def forward(self, x):
#         latent = self.encoder(x)
#         reconstructed = self.decoder(latent)
#         return reconstructed

# class ML_IPS_Controller(app_manager.RyuApp):
#     OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

#     def __init__(self, *args, **kwargs):
#         super(ML_IPS_Controller, self).__init__(*args, **kwargs)
#         self.mac_to_port = {}
#         self.datapaths = {}
        
#         # 1. Tải các mô hình ML [cite: 2729, 3239]
#         self.scaler = joblib.load('models/scaler.joblib')
#         self.rf_model = joblib.load('models/rf_classifier.joblib')
        
#         # Load Autoencoder (Pytorch)
#         input_dim = 30
#         latent_dim = 5
#         hidden_layers = [22, 12]
#         self.ae = DeepAutoencoder(input_dim, latent_dim, hidden_layers)
#         self.ae.load_state_dict(torch.load('models/autoencoder.pth'))
#         self.ae.eval()

#         # Danh sách 30 đặc trưng mRMR từ Giai đoạn 1 [cite: 3239]
#         self.mrmr_features = [
#             'bidirectional_rst_packets', 'src2dst_bytes', 'dst2src_iat_min', 
#             'bidirectional_ece_packets', 'src2dst_packets', 'bidirectional_idle_std',
#             # ... Thêm đủ 30 tên cột tương ứng từ NFStreamer sang CIC-IDS
#         ]

#         # Khởi chạy luồng bắt traffic bằng NFStreamer
#         self.monitor_thread = hub.spawn(self._traffic_monitor)

#     def _traffic_monitor(self):
#         """Tiến trình trích xuất đặc trưng realtime sử dụng NFStreamer"""
#         # Lưu ý: 'any' hoặc tên interface cụ thể trong Mininet (vd: 's1-eth1')
#         streamer = NFStreamer(source="any", promiscuous_mode=True)
        
#         for flow in streamer:
#             # Chuyển đổi flow thành định dạng dữ liệu có thể predict
#             self._process_flow(flow)

#     def _process_flow(self, flow):
#         """Xử lý từng luồng dữ liệu và đưa ra dự đoán"""
#         try:
#             # Trích xuất 30 đặc trưng (cần map chính xác từ nfstream flow sang mrmr_features) [cite: 2504, 3239]
#             # Ví dụ minh họa:
#             data = {
#                 'bidirectional_rst_packets': flow.bidirectional_rst_packets,
#                 'src2dst_bytes': flow.src2dst_bytes,
#                 # ... Map đủ 30 đặc trưng
#             }
#             df = pd.DataFrame([data])
            
#             # Chuẩn hóa dữ liệu [cite: 2511]
#             X_scaled = self.scaler.transform(df)
#             X_tensor = torch.FloatTensor(X_scaled)
            
#             # Trích xuất đặc trưng ẩn qua Autoencoder (5 chiều) [cite: 2592, 2601]
#             with torch.no_grad():
#                 X_latent = self.ae.encoder(X_tensor).numpy()
            
#             # Hợp nhất đặc trưng: 20 mRMR + 5 latent = 25 dims [cite: 2607]
#             X_fusion = np.hstack([X_scaled[:, :20], X_latent])
            
#             # Dự đoán [cite: 2620]
#             prediction = self.rf_model.predict(X_fusion)[0]
            
#             if prediction == 1: # Nhãn ATTACK [cite: 2034]
#                 self.logger.info("--- PHÁT HIỆN TẤN CÔNG! IP nguồn: %s ---", flow.src_ip)
#                 self._block_attacker(flow.src_ip)
                
#         except Exception as e:
#             self.logger.error("Lỗi xử lý luồng: %s", e)

#     def _block_attacker(self, src_ip):
#         """Tự động cài flow rule để chặn IP tấn công (IPS) [cite: 3501]"""
#         for dp in self.datapaths.values():
#             parser = dp.ofproto_parser
#             # Khớp gói tin IPv4 có địa chỉ nguồn là kẻ tấn công
#             match = parser.OFPMatch(eth_type=0x0800, ipv4_src=src_ip)
#             # Không có Action đồng nghĩa với việc DROP gói tin
#             self.add_flow(dp, 100, match, [], idle_timeout=300)
#             self.logger.info("Đã chặn lưu lượng từ IP %s tại Switch %d", src_ip, dp.id)

#     def add_flow(self, datapath, priority, match, actions, buffer_id=None, idle_timeout=0):
#         ofproto = datapath.ofproto
#         parser = datapath.ofproto_parser

#         inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
#         if buffer_id:
#             mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
#                                     priority=priority, match=match,
#                                     instructions=inst, idle_timeout=idle_timeout)
#         else:
#             mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
#                                     match=match, instructions=inst, 
#                                     idle_timeout=idle_timeout)
#         datapath.send_msg(mod)

#     @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
#     def switch_features_handler(self, ev):
#         datapath = ev.msg.datapath
#         self.datapaths[datapath.id] = datapath
#         # ... logic nguyên bản của simple_switch_stp_13.py ...

# import torch
# import torch.nn as nn
# import joblib
# import numpy as np
# import pandas as pd
# from nfstream import NFStreamer
# from ryu.base import app_manager
# from ryu.controller import ofp_event
# from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
# from ryu.controller.handler import set_ev_cls
# from ryu.ofproto import ofproto_v1_3
# from ryu.lib import hub
# from ryu.lib.packet import packet, ethernet, ipv4
# from ryu.app import simple_switch_13

# # --- Định nghĩa kiến trúc Autoencoder từ Giai đoạn 1 ---
# class DeepAutoencoder(nn.Module):
#     def __init__(self, input_dim=30, latent_dim=5, hidden_layers=[22, 12]):
#         super(DeepAutoencoder, self).__init__()
#         encoder_modules = []
#         current_dim = input_dim
#         for h_dim in hidden_layers:
#             encoder_modules.append(nn.Linear(current_dim, h_dim))
#             encoder_modules.append(nn.BatchNorm1d(h_dim))
#             encoder_modules.append(nn.LeakyReLU(0.2))
#             current_dim = h_dim
#         encoder_modules.append(nn.Linear(current_dim, latent_dim))
#         self.encoder = nn.Sequential(*encoder_modules)

#     def forward(self, x):
#         return self.encoder(x)

# class ML_IPS_Controller(simple_switch_13.SimpleSwitch13):
#     OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

#     def __init__(self, *args, **kwargs):
#         super(ML_IPS_Controller, self).__init__(*args, **kwargs)
#         self.datapaths = {}
        
#         # 1. Tải mô hình và cấu hình (Đảm bảo các file nằm trong thư mục model/)
#         try:
#             self.scaler = joblib.load('model/scaler.joblib')
#             self.rf_model = joblib.load('model/rf_model.pkl')
            
#             # Khởi tạo AE và nạp trọng số
#             self.ae = DeepAutoencoder(input_dim=30, latent_dim=5, hidden_layers=[22, 12])
#             self.ae.load_state_dict(torch.load('model/autoencoder.pth', map_location=torch.device('cpu')))
#             self.ae.eval()
#             self.logger.info("✅ Đã nạp thành công Scaler, Autoencoder và Random Forest.")
#         except Exception as e:
#             self.logger.error("❌ Lỗi khi nạp mô hình: %s", e)

#         # Danh sách 30 đặc trưng mRMR (Thứ tự phải khớp tuyệt đối với lúc train)
#         self.FEATURE_30 = [
#             'RST Flag Count', 'Total Length of Fwd Packets', 'Bwd IAT Min', 'ECE Flag Count',
#             'act_data_pkt_fwd', 'Idle Std', 'Bwd Packet Length Min', 'Total Fwd Packets',
#             'Bwd IAT Mean', 'PSH Flag Count', 'Destination Port', 'Flow IAT Std',
#             'Bwd Packet Length Std', 'Bwd IAT Max', 'Fwd Packet Length Max', 'Fwd PSH Flags',
#             'Active Min', 'Init_Win_bytes_backward', 'SYN Flag Count', 'Flow Duration',
#             'Fwd IAT Min', 'Down/Up Ratio', 'Bwd IAT Std', 'Fwd Packet Length Std',
#             'Fwd IAT Total', 'Bwd Packets/s', 'Active Mean', 'Fwd IAT Mean', 'URG Flag Count',
#             'Min Packet Length'
#         ]

#         # 2. Khởi chạy luồng giám sát Traffic bằng NFStreamer
#         self.monitor_thread = hub.spawn(self._traffic_monitor)

#     def _traffic_monitor(self):
#         """Tiến trình trích xuất đặc trưng realtime"""
#         self.logger.info("[*] Đang bắt đầu giám sát luồng dữ liệu trực tuyến...")
#         # Lắng nghe trên interface s1-eth10 (kết nối gateway trong topo_2.py)
#         streamer = NFStreamer(source="s1-eth10", statistical_analysis=True, idle_timeout=5)
        
#         for flow in streamer:
#             # Chỉ xử lý các luồng TCP/UDP có dữ liệu
#             if flow.protocol in [6, 17] and flow.bidirectional_packets > 0:
#                 self._inference_pipeline(flow)

#     def _inference_pipeline(self, flow):
#         """Quy trình Mapping -> AE -> Fusion -> Prediction"""
#         try:
#             # Bước A: Ánh xạ từ NFStreamer sang 30 đặc trưng CIC-IDS
#             # (Tạm thời bỏ qua hiệu suất trích xuất theo yêu cầu của bạn)
#             duration = (flow.bidirectional_duration_ms / 1000.0) if flow.bidirectional_duration_ms > 0 else 0.0001
            
#             raw_features = {
#                 'Total Length of Fwd Packets': flow.src2dst_bytes,
#                 'Bwd IAT Min': flow.dst2src_min_piat_ms,
#                 'act_data_pkt_fwd': flow.src2dst_packets,
#                 'Bwd Packet Length Min': flow.dst2src_min_ps,
#                 'Total Fwd Packets': flow.src2dst_packets,
#                 'Bwd IAT Mean': flow.dst2src_mean_piat_ms,
#                 'PSH Flag Count': flow.src2dst_psh_packets + flow.dst2src_psh_packets,
#                 'Destination Port': flow.dst_port,
#                 'Flow IAT Std': flow.bidirectional_stddev_piat_ms,
#                 'Bwd Packet Length Std': flow.dst2src_stddev_ps,
#                 'Bwd IAT Max': flow.dst2src_max_piat_ms,
#                 'Fwd Packet Length Max': flow.src2dst_max_ps,
#                 'Fwd PSH Flags': flow.src2dst_psh_packets,
#                 'SYN Flag Count': getattr(flow, 'src2dst_syn_packets', 0),
#                 'Flow Duration': flow.bidirectional_duration_ms,
#                 'Fwd IAT Min': flow.src2dst_min_piat_ms,
#                 'Down/Up Ratio': flow.dst2src_packets / (flow.src2dst_packets + 1),
#                 'Bwd IAT Std': flow.dst2src_stddev_piat_ms,
#                 'Fwd Packet Length Std': flow.src2dst_stddev_ps,
#                 'Fwd IAT Total': flow.src2dst_duration_ms,
#                 'Bwd Packets/s': flow.dst2src_packets / duration,
#                 'Fwd IAT Mean': flow.src2dst_mean_piat_ms,
#                 'URG Flag Count': flow.src2dst_urg_packets + flow.dst2src_urg_packets,
#                 'Min Packet Length': flow.bidirectional_min_ps,
#                 # Điền 0 cho các đặc trưng chưa trích xuất được trực tiếp
#                 'RST Flag Count': 0, 'ECE Flag Count': 0, 'Idle Std': 0, 
#                 'Active Min': 0, 'Active Mean': 0, 'Init_Win_bytes_backward': 0
#             }

#             # Đưa về dạng DataFrame để Scaler xử lý
#             ordered_values = [raw_features.get(f, 0) for f in self.FEATURE_30]
#             df_input = pd.DataFrame([ordered_values], columns=self.FEATURE_30)

#             # Bước B: Xử lý qua mô hình
#             X_scaled = self.scaler.transform(df_input)
#             X_tensor = torch.FloatTensor(X_scaled)
            
#             with torch.no_grad():
#                 # Lấy 5 đặc trưng ẩn từ Encoder
#                 X_latent = self.ae.encoder(X_tensor).numpy()
            
#             # Kết hợp 20 đặc trưng gốc đầu tiên + 5 đặc trưng ẩn = 25 chiều
#             X_fusion = np.hstack([X_scaled[:, :20], X_latent])
            
#             # Dự đoán nhãn
#             prediction = self.rf_model.predict(X_fusion)[0]

#             if prediction == 1: # Giả định 1 là nhãn ATTACK
#                 self.logger.warning("🚨 CẢNH BÁO: Phát hiện tấn công từ IP %s", flow.src_ip)
#                 self._apply_ips_policy(flow.src_ip)

#         except Exception as e:
#             self.logger.error("⚠️ Lỗi trong quá trình suy luận: %s", e)

#     def _apply_ips_policy(self, attacker_ip):
#         """Cơ chế IPS: Chặn lưu lượng từ kẻ tấn công bằng Flow Rule"""
#         for dp in self.datapaths.values():
#             parser = dp.ofproto_parser
#             # Khớp (match) gói tin dựa trên IP nguồn
#             match = parser.OFPMatch(eth_type=0x0800, ipv4_src=attacker_ip)
#             # Action trống có nghĩa là DROP gói tin
#             actions = []
#             # Cài đặt flow rule với ưu tiên cao (priority 100) và thời gian sống (idle_timeout)
#             self.add_flow(dp, 100, match, actions, idle_timeout=300)
#             self.logger.info("⛔ Đã thực thi lệnh chặn IP: %s trên Switch %d", attacker_ip, dp.id)

#     @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
#     def switch_features_handler(self, ev):
#         datapath = ev.msg.datapath
#         self.datapaths[datapath.id] = datapath
#         super(ML_IPS_Controller, self).switch_features_handler(ev)

#     def add_flow(self, datapath, priority, match, actions, buffer_id=None, idle_timeout=0):
#         ofproto = datapath.ofproto
#         parser = datapath.ofproto_parser
#         inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        
#         mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
#                                 match=match, instructions=inst,
#                                 buffer_id=buffer_id or ofproto.OFP_NO_BUFFER,
#                                 idle_timeout=idle_timeout)
#         datapath.send_msg(mod)

import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
from nfstream import NFStreamer
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub
from ryu.lib.packet import packet, ethernet, ipv4
from ryu.app import simple_switch_13

# --- 1. Định nghĩa kiến trúc Autoencoder (Bắt buộc phải khớp với Phase 1) ---
class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim=30, latent_dim=5, hidden_layers=[22, 12]):
        super(DeepAutoencoder, self).__init__()
        # Encoder
        encoder_layers = []
        last_dim = input_dim
        for h_dim in hidden_layers:
            encoder_layers.append(nn.Linear(last_dim, h_dim))
            encoder_layers.append(nn.BatchNorm1d(h_dim))
            encoder_layers.append(nn.LeakyReLU(0.2))
            last_dim = h_dim
        encoder_layers.append(nn.Linear(last_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x):
        return self.encoder(x)

# --- 2. Controller Ryu tích hợp IDS/IPS ML ---
class ML_IPS_Controller(simple_switch_13.SimpleSwitch13):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(ML_IPS_Controller, self).__init__(*args, **kwargs)
        self.datapaths = {}
        
        # Đường dẫn tới các Model (Đảm bảo các file này tồn tại trong thư mục model/)
        self.PATH_SCALER = 'model/scaler.joblib'
        self.PATH_RF = 'model/rf_model.pkl'
        self.PATH_AE = 'model/autoencoder.pth'

        # Load Models
        try:
            self.scaler = joblib.load(self.PATH_SCALER)
            self.rf_model = joblib.load(self.PATH_RF)
            self.ae = DeepAutoencoder(input_dim=30, latent_dim=5)
            self.ae.load_state_dict(torch.load(self.PATH_AE, map_location=torch.device('cpu')))
            self.ae.eval()
            self.logger.info("✅ Hệ thống ML: Đã sẵn sàng (Scaler, AE, RF).")
        except Exception as e:
            self.logger.error("❌ Lỗi Load Model: %s", e)

        # Thứ tự 30 đặc trưng mRMR (Phải khớp chính xác với thứ tự lúc train)
        self.MRMR_RANK_30 = [
            'RST Flag Count', 'Total Length of Fwd Packets', 'Bwd IAT Min', 'ECE Flag Count',
            'act_data_pkt_fwd', 'Idle Std', 'Bwd Packet Length Min', 'Total Fwd Packets',
            'Bwd IAT Mean', 'PSH Flag Count', 'Destination Port', 'Flow IAT Std',
            'Bwd Packet Length Std', 'Bwd IAT Max', 'Active Min', 'Fwd PSH Flags',
            'Fwd Packet Length Max', 'Init_Win_bytes_backward', 'Flow Duration',
            'SYN Flag Count', 'Fwd IAT Min', 'Bwd IAT Std', 'Down/Up Ratio',
            'Fwd Header Length', 'Fwd IAT Total', 'Active Mean', 'Fwd Packet Length Std',
            'Fwd IAT Mean', 'URG Flag Count', 'Min Packet Length'
        ]

        # Khởi chạy luồng giám sát Traffic Online
        self.monitor_thread = hub.spawn(self._traffic_monitor)

    def _traffic_monitor(self):
        """Module trích xuất đặc trưng Real-time bằng NFStreamer"""
        self.logger.info("[*] Đang lắng nghe traffic trên interface s1-eth10...")
        # statistical_analysis=True để lấy các thông số IAT, Std, Min, Max
        streamer = NFStreamer(source="s1-eth10", statistical_analysis=True, idle_timeout=5)
        
        for flow in streamer:
            # Lọc traffic: Chỉ phân tích TCP/UDP và các flow đã kết thúc hoặc timeout
            if flow.protocol in [6, 17] and flow.bidirectional_packets > 0:
                self._run_inference(flow)

    def _run_inference(self, flow):
        """Quy trình: Mapping -> Preprocessing -> AE -> RF -> IPS"""
        try:
            # 1. MAPPING (Ánh xạ từ NFStreamer sang CIC-IDS Features)
            # Tính toán các giá trị phái sinh
            duration_sec = (flow.bidirectional_duration_ms / 1000.0) if flow.bidirectional_duration_ms > 0 else 0.0001
            
            mapping = {
                'RST Flag Count': flow.src2dst_rst_packets + flow.dst2src_rst_packets,
                'Total Length of Fwd Packets': flow.src2dst_bytes,
                'Bwd IAT Min': flow.dst2src_min_piat_ms,
                'ECE Flag Count': flow.src2dst_ece_packets + flow.dst2src_ece_packets,
                'act_data_pkt_fwd': flow.src2dst_packets, # Mapping xấp xỉ
                'Idle Std': 0, # Chưa hỗ trợ trực tiếp
                'Bwd Packet Length Min': flow.dst2src_min_ps,
                'Total Fwd Packets': flow.src2dst_packets,
                'Bwd IAT Mean': flow.dst2src_mean_piat_ms,
                'PSH Flag Count': flow.src2dst_psh_packets + flow.dst2src_psh_packets,
                'Destination Port': flow.dst_port,
                'Flow IAT Std': flow.bidirectional_stddev_piat_ms,
                'Bwd Packet Length Std': flow.dst2src_stddev_ps,
                'Bwd IAT Max': flow.dst2src_max_piat_ms,
                'Active Min': 0, # Chưa hỗ trợ trực tiếp
                'Fwd PSH Flags': flow.src2dst_psh_packets,
                'Fwd Packet Length Max': flow.src2dst_max_ps,
                'Init_Win_bytes_backward': 0, # Yêu cầu deep packet inspection
                'Flow Duration': flow.bidirectional_duration_ms,
                'SYN Flag Count': getattr(flow, 'src2dst_syn_packets', 0),
                'Fwd IAT Min': flow.src2dst_min_piat_ms,
                'Bwd IAT Std': flow.dst2src_stddev_piat_ms,
                'Down/Up Ratio': flow.dst2src_packets / (flow.src2dst_packets + 1),
                'Fwd Header Length': flow.src2dst_header_bytes,
                'Fwd IAT Total': flow.src2dst_duration_ms,
                'Active Mean': 0, # Chưa hỗ trợ trực tiếp
                'Fwd Packet Length Std': flow.src2dst_stddev_ps,
                'Fwd IAT Mean': flow.src2dst_mean_piat_ms,
                'URG Flag Count': flow.src2dst_urg_packets + flow.dst2src_urg_packets,
                'Min Packet Length': flow.bidirectional_min_ps
            }

            # 2. PREPROCESSING: Sắp xếp đúng thứ tự mRMR
            features_vector = [mapping.get(f, 0) for f in self.MRMR_RANK_30]
            df_input = pd.DataFrame([features_vector], columns=self.MRMR_RANK_30)

            # Scaling
            X_scaled = self.scaler.transform(df_input)
            X_tensor = torch.FloatTensor(X_scaled)

            # 3. AUTOENCODER: Trích xuất 5 đặc trưng ẩn (Latent Space)
            with torch.no_grad():
                X_latent = self.ae.encoder(X_tensor).numpy()

            # 4. FUSION: Kết hợp Top-20 gốc + 5 Latent = 25 dimensions
            X_fusion = np.hstack([X_scaled[:, :20], X_latent])

            # 5. PREDICTION
            prediction = self.rf_model.predict(X_fusion)[0]

            # 6. IPS ACTION: Nếu là ATTACK (Giả định nhãn 1 là Attack)
            if prediction == 1:
                self.logger.warning("🚨 [ATTACK DETECTED] Source IP: %s | Type: Malicious Flow", flow.src_ip)
                self._block_attacker(flow.src_ip)
            else:
                self.logger.info("🟢 [NORMAL] %s -> %s", flow.src_ip, flow.dst_ip)

        except Exception as e:
            self.logger.error("⚠️ Lỗi Inference: %s", e)

    def _block_attacker(self, attacker_ip):
        """Cài đặt Flow Rule để chặn IP kẻ tấn công (DROP)"""
        for dp in self.datapaths.values():
            parser = dp.ofproto_parser
            # Match gói tin IPv4 có địa chỉ nguồn là attacker_ip
            match = parser.OFPMatch(eth_type=0x0800, ipv4_src=attacker_ip)
            # Actions trống = DROP gói tin
            actions = []
            # Ưu tiên cao hơn các rule switching bình thường (priority 100)
            # Tự động xóa rule sau 300 giây (5 phút) để tránh tràn bảng flow
            self.add_flow(dp, 100, match, actions, idle_timeout=300)
            self.logger.info("⛔ Đã thực thi DROP flow từ IP: %s trên Switch %d", attacker_ip, dp.id)

    # --- HÀM BỔ TRỢ ---
    def add_flow(self, datapath, priority, match, actions, buffer_id=None, idle_timeout=0):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                    priority=priority, match=match,
                                    instructions=inst, idle_timeout=idle_timeout)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                    match=match, instructions=inst, 
                                    idle_timeout=idle_timeout)
        datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        self.datapaths[datapath.id] = datapath
        super(ML_IPS_Controller, self).switch_features_handler(ev)