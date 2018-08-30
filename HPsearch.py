import train_model
from itertools import product
from HParams import global_hparams as gh
import sys
import json
import os
import shutil
from io_utils import filename_from_path
TEMP_CONF_DIR = "/temp_conf"
from random import shuffle
from io_utils import safe_delete_dir

def main():
    assert len(sys.argv) >= 2
    conf_path = sys.argv[1]
    conf_name = filename_from_path(conf_path)
    with open(conf_path, 'r') as f:
        conf_dict = json.load(f)
    alphas = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4] or [gh.alpha]
    discounts = [0.9, 0.85, 0.8] or [gh.discount]
    epsilons = [0.001, 0.003, 0.01, 0.03] or [gh.epsilon]
    num_exps = 20
    exps = list(product(alphas, discounts, epsilons))
    shuffle(exps)
    exps = exps[:num_exps]
    safe_delete_dir(TEMP_CONF_DIR)
    os.mkdir(TEMP_CONF_DIR)
    for alpha, discount, epsilon in exps:
        conf_dict["alpha"] = alpha
        conf_dict["discount"] = discount
        conf_dict["epsilon"] = epsilon
        gh.set_params(conf_dict)
        cur_path = os.path.join(TEMP_CONF_DIR,
                               conf_name+ "_a_{}_d_{}_e_{}".format(alpha, discount, epsilon) + ".json")
        with open(cur_path, "w") as f:
            json.dump(conf_dict, f, indent=2)
        train_model.run_single_model(cur_path)
    shutil.rmtree(TEMP_CONF_DIR)




if __name__ == "__main__":
    main()

