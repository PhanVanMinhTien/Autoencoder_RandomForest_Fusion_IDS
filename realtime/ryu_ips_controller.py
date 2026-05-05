# ryu_ips_controller.py

from ryu.app import simple_switch_13
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, DEAD_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.app.wsgi import ControllerBase, WSGIApplication, route

from webob import Response
import json


IPS_INSTANCE_NAME = "ips_app"


class RyuIPSMitigationController(simple_switch_13.SimpleSwitch13):
    """
    Ryu controller:
    - Handles basic L2 forwarding using simple_switch_13.
    - Exposes REST API for external ML predictor.
    - Installs OpenFlow DROP rules when malicious source IP is reported.
    """

    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS = {
        "wsgi": WSGIApplication
    }

    def __init__(self, *args, **kwargs):
        super(RyuIPSMitigationController, self).__init__(*args, **kwargs)

        self.datapaths = {}
        self.blocked_ips = set()

        wsgi = kwargs["wsgi"]
        wsgi.register(IPSMitigationAPI, {IPS_INSTANCE_NAME: self})

        self.logger.info("Ryu IPS Mitigation Controller started.")
        self.logger.info("REST API available: POST /block")

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        self.datapaths[datapath.id] = datapath

        self.logger.info("Switch connected: dpid=%s", datapath.id)

        # Keep normal simple_switch behavior
        super(RyuIPSMitigationController, self).switch_features_handler(ev)

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def state_change_handler(self, ev):
        datapath = ev.datapath

        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.datapaths[datapath.id] = datapath
                self.logger.info("Registered datapath: %s", datapath.id)

        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                del self.datapaths[datapath.id]
                self.logger.info("Unregistered datapath: %s", datapath.id)

    def block_ip(self, src_ip, idle_timeout=300):
        """
        Install DROP rule for IPv4 packets from src_ip on all connected switches.
        """
        if not src_ip:
            return False, "src_ip is empty"

        self.blocked_ips.add(src_ip)

        for dp in self.datapaths.values():
            ofproto = dp.ofproto
            parser = dp.ofproto_parser

            match = parser.OFPMatch(
                eth_type=0x0800,
                ipv4_src=src_ip
            )

            # Empty action list means DROP
            actions = []
            inst = [
                parser.OFPInstructionActions(
                    ofproto.OFPIT_APPLY_ACTIONS,
                    actions
                )
            ]

            mod = parser.OFPFlowMod(
                datapath=dp,
                priority=100,
                match=match,
                instructions=inst,
                idle_timeout=idle_timeout
            )

            dp.send_msg(mod)

            self.logger.warning(
                "Installed DROP rule: src_ip=%s on switch=%s idle_timeout=%s",
                src_ip,
                dp.id,
                idle_timeout
            )

        return True, f"Blocked {src_ip}"

    def unblock_ip(self, src_ip):
        """
        Delete DROP rule for IPv4 packets from src_ip.
        """
        if not src_ip:
            return False, "src_ip is empty"

        if src_ip in self.blocked_ips:
            self.blocked_ips.remove(src_ip)

        for dp in self.datapaths.values():
            ofproto = dp.ofproto
            parser = dp.ofproto_parser

            match = parser.OFPMatch(
                eth_type=0x0800,
                ipv4_src=src_ip
            )

            mod = parser.OFPFlowMod(
                datapath=dp,
                command=ofproto.OFPFC_DELETE,
                out_port=ofproto.OFPP_ANY,
                out_group=ofproto.OFPG_ANY,
                priority=100,
                match=match
            )

            dp.send_msg(mod)

            self.logger.info(
                "Deleted DROP rule: src_ip=%s on switch=%s",
                src_ip,
                dp.id
            )

        return True, f"Unblocked {src_ip}"


class IPSMitigationAPI(ControllerBase):
    def __init__(self, req, link, data, **config):
        super(IPSMitigationAPI, self).__init__(req, link, data, **config)
        self.ips_app = data[IPS_INSTANCE_NAME]

    @route("ips", "/block", methods=["POST"])
    def block(self, req, **kwargs):
        try:
            body = req.json if req.body else {}
            src_ip = body.get("src_ip")
            idle_timeout = int(body.get("idle_timeout", 300))

            ok, message = self.ips_app.block_ip(src_ip, idle_timeout)

            status = 200 if ok else 400
            return Response(
                status=status,
                content_type="application/json",
                body=json.dumps({
                    "ok": ok,
                    "message": message,
                    "blocked_ips": list(self.ips_app.blocked_ips)
                }).encode("utf-8")
            )

        except Exception as e:
            return Response(
                status=500,
                content_type="application/json",
                body=json.dumps({
                    "ok": False,
                    "error": str(e)
                }).encode("utf-8")
            )

    @route("ips", "/unblock", methods=["POST"])
    def unblock(self, req, **kwargs):
        try:
            body = req.json if req.body else {}
            src_ip = body.get("src_ip")

            ok, message = self.ips_app.unblock_ip(src_ip)

            status = 200 if ok else 400
            return Response(
                status=status,
                content_type="application/json",
                body=json.dumps({
                    "ok": ok,
                    "message": message,
                    "blocked_ips": list(self.ips_app.blocked_ips)
                }).encode("utf-8")
            )

        except Exception as e:
            return Response(
                status=500,
                content_type="application/json",
                body=json.dumps({
                    "ok": False,
                    "error": str(e)
                }).encode("utf-8")
            )

    @route("ips", "/status", methods=["GET"])
    def status(self, req, **kwargs):
        return Response(
            status=200,
            content_type="application/json",
            body=json.dumps({
                "ok": True,
                "connected_switches": list(self.ips_app.datapaths.keys()),
                "blocked_ips": list(self.ips_app.blocked_ips)
            }).encode("utf-8")
        )