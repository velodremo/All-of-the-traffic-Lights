
from HParams import global_hparams
from junction import Junction
import  sumo_backend


def build_junctions():
    j_ids = global_hparams.junctions
    backend = sumo_backend.get_backend()
    junctions = [Junction(j_id, j_id, backend) for j_id in j_ids] # TODO: compute junction neighbors
    return junctions

