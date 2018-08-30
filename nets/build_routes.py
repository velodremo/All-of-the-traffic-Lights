from xml.etree import ElementTree as ET
from itertools import combinations
VEHICLES = {"car": {
    "accel":"1.0",
    "decel":"0.5",
    "id":"car",
    "length":"2.0",
    "maxSpeed": "27.0",
    "sigma": "0.0"
},
    "bus": {
    "accel":"0.8",
    "decel":"0.7",
    "id":"bus",
    "length":"12.0",
    "maxSpeed": "25.0",
    "sigma": "0.0"
},
    "ambulance": {
    "accel":"1.0",
    "decel":"0.5",
    "id":"ambulance",
    "length":"6.0",
    "maxSpeed": "30.5",
    "sigma": "0.0"
}
}
V_PROBS = {
    "car": "$car",
    "ambulance": "$ambulance",
    "bus": "$bus"
}
nodes = ["0","1",
         "0u", "0ur", "0ul",
         "0l", "0lu", "0ld",
         "0d", "0dl", "0dr",
         "1u", "1ur", "1ul",
         "1d", "1dl", "1dr",
         "1r", "1ru", "1rd"]

STRAIGHT_LEGAL = {"0ul_0ur", "0ul_0dl", "0ul_0dr", "0ur_0ul","0ur_0dl", "0ur_0dr","0lu_0ld","0lu_1ru",
                  "0lu_1rd","0ld_0lu","0ld_1ru",
                  "0ld_1rd", "0dl_0dr", "0dl_0ul", "0dl_0ur", "0dr_0dl", "0dr_0ul", "0dr_0ur",

                  "1ul_1ur", "1ul_1dl", "1ul_1dr", "1ur_1ul", "1ur_1dl", "1ur_1dr", "1ru_1rd", "1ru_0lu",
                  "1ru_0ld", "1rd_1ru", "1rd_0lu",
                  "1rd_0ld", "1dl_1dr", "1dl_1ul", "1dl_1ur", "1dr_1dl", "1dr_1ul", "1dr_1ur"
                  }
STRAIGHT_LEGAL = {"r_" + name for name in STRAIGHT_LEGAL}
class Node():
    def __init__(self, name, neighbors):
        self.neighbors = neighbors
        self.name = name


NODES = []
for node in nodes:
    NODES.append(Node(node, []))


for n1, n2 in combinations(NODES, 2):
    if len(n1.name) == len(n2.name) + 1 and n1.name[:-1] == n2.name:
        n1.neighbors.append(n2)
        n2.neighbors.append(n1)
    if len(n1.name) == len(n2.name) - 1 and n2.name[:-1] == n1.name:
        n1.neighbors.append(n2)
        n2.neighbors.append(n1)
    if n1.name in ["0", "1"] and n2.name in ["0", "1"]:
        n1.neighbors.append(n2)
        n2.neighbors.append(n1)

end_nodes = list(filter(lambda n: len(n.name) == 3, NODES))
routes = []


def generate_routes(node, blacklist, cur_route):
    cur_route.append(node)
    if node in end_nodes and len(cur_route) > 1:
        routes.append(cur_route)
    else:
        for ne in node.neighbors:
            if ne not in blacklist:
                generate_routes(ne, [node], cur_route.copy())



for node in end_nodes:
    generate_routes(node,[],[])

routes_edges = {}
for route in routes:
    cur_route_edges = []
    for i in range(len(route) - 1):
        cur_route_edges.append(route[i].name + "_" + route[i+1].name)
    r_name = "r_"+ route[0].name + "_" + route[-1].name
    routes_edges[r_name]= " ".join(cur_route_edges)

root = ET.Element("routes")
for k, vehicle in VEHICLES.items():
    vtype = ET.SubElement(root, "vType", vehicle)
for r_name, route in routes_edges.items():
    if r_name in STRAIGHT_LEGAL:
        ET.SubElement(root, "route",{"id": r_name, "edges": route})
    # '    <flow depart="1" id="flow_w_e" route="route_we" type="Car" begin="0" end="50000" probability="$we" />'
for r_name, route in routes_edges.items():
    if r_name in STRAIGHT_LEGAL:
        for v_name, vehicle in VEHICLES.items():
            ET.SubElement(root, "flow", {"depart": "1", "id": "flow_"+r_name+"_"+v_name, "route": r_name,
                                         "type": vehicle["id"], "begin": "0", "end": "$end", "probability":
                                             V_PROBS[v_name]})

mydata = ET.tostring(root)
print(mydata)
with open("traffic.rou.xml", "wb") as f:
    f.write(mydata)








