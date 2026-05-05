from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch, Host
from mininet.cli import CLI
from mininet.log import setLogLevel

def enterprise_topo():
    net = Mininet(topo=None, build=True, ipBase='192.168.0.0/16')

    print("*** Adding Controller (ML-based IDS/IPS)")
    c0 = net.addController(name='c0', controller=RemoteController, ip='127.0.0.1', port=6653)

    print("*** Adding Edge Firewall & Core Switch")
    # s1 đóng vai trò Core Switch - Nơi tập trung lưu lượng để ML phân tích
    core_sw = net.addSwitch('s1', cls=OVSSwitch, protocols='OpenFlow13')
    
    # Nút fw đóng vai trò Edge Firewall/Router lớp 3
    fw = net.addHost('fw', ip='10.0.0.1/24') 

    print("*** Adding Access Switches for Departments")
    s3 = net.addSwitch('s3', cls=OVSSwitch, protocols='OpenFlow13') # Sale
    s4 = net.addSwitch('s4', cls=OVSSwitch, protocols='OpenFlow13') # HR
    s5 = net.addSwitch('s5', cls=OVSSwitch, protocols='OpenFlow13') # IT
    s6 = net.addSwitch('s6', cls=OVSSwitch, protocols='OpenFlow13') # Server Farm (DMZ)

    print("*** Adding External Host (Internet Attacker)")
    attacker = net.addHost('attacker', ip='10.0.0.100/24', defaultRoute='via 10.0.0.1')

    print("*** Creating Links")
    # Kết nối Internet vào Firewall
    net.addLink(attacker, fw)
    # Kết nối Firewall vào Core Switch (Interface s1-eth10 dùng để monitor)
    net.addLink(fw, core_sw, port2=10) 

    # Kết nối Core Switch tới các Access Switches
    net.addLink(core_sw, s3)
    net.addLink(core_sw, s4)
    net.addLink(core_sw, s5)
    net.addLink(core_sw, s6)

    # Thêm các host cho từng phòng ban (Ví dụ mỗi vùng 1 host đại diện)
    h_sale = net.addHost('h_sale', ip='192.168.10.10/24', defaultRoute='via 192.168.10.1')
    h_hr   = net.addHost('h_hr',   ip='192.168.20.10/24', defaultRoute='via 192.168.20.1')
    h_it   = net.addHost('h_it',   ip='192.168.30.10/24', defaultRoute='via 192.168.30.1')
    h_srv  = net.addHost('h_srv',  ip='192.168.100.10/24', defaultRoute='via 192.168.100.1')

    net.addLink(h_sale, s3)
    net.addLink(h_hr, s4)
    net.addLink(h_it, s5)
    net.addLink(h_srv, s6)

    print("*** Starting Network")
    net.start()

    # Cấu hình IP Forwarding và Route cho Firewall nút 'fw'
    fw_node = net.get('fw')
    fw_node.cmd('sysctl -w net.ipv4.ip_forward=1')
    # Cấu hình các sub-interface cho Firewall (Inter-VLAN Routing)
    fw_node.cmd('ifconfig fw-eth1 192.168.10.1/24')
    fw_node.cmd('ifconfig fw-eth1:1 192.168.20.1/24')
    fw_node.cmd('ifconfig fw-eth1:2 192.168.30.1/24')
    fw_node.cmd('ifconfig fw-eth1:3 192.168.100.1/24')

    CLI(net)
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    enterprise_topo()