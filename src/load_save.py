import json
import torch
import numpy as np
from src.model import BuildNetwork_previous, BuildNetwork, BuildNetworkNew

path = r"D:\Emilien\Documents\Cours\Master_Thesis\Harvard_Master_Thesis\model_history/"


def save_model(trained_model, formatted_datetime_int, equation_name, name,
               x_range, iterations, hid_lay, num_equations, num_heads, A_list,
               v_list, force_list, alpha_list, loss_hist):
    # Save model history
    path_equation = path + f"{equation_name}/"
    torch.save(trained_model.state_dict(), path_equation + f"{name}_{formatted_datetime_int}")
    history = {}
    history["x_range"] = x_range
    history["iterations"] = iterations
    history["hid_lay"] = [int(i) for i in hid_lay]
    history["num_equations"] = num_equations
    history["num_heads"] = num_heads

    history["A"] = [A.cpu().numpy().tolist() for A in A_list]
    history["v"] = [v.cpu().numpy().tolist() for v in v_list]
    if callable(force_list[0]):
        history["force"] = None
    else:
        history["force"] = [f.cpu().numpy().tolist() for f in force_list]
    history["alpha_list"] = alpha_list if isinstance(alpha_list, list) else alpha_list.tolist()

    loss_hist["head"] = loss_hist["head"] if isinstance(loss_hist["head"], list) else loss_hist["head"].tolist()
    history["loss_hist"] = loss_hist

    with open(path_equation + f"history_{name}_{formatted_datetime_int}.json", "w") as fp:
        print(path_equation + f"history_{name}_{formatted_datetime_int}.json")
        json.dump(history, fp)


def load_run_history(equation_name, model_file, device, prev):
    path_equation = path + f"{equation_name}/"
    with open(path_equation + "history_" + str(model_file) + ".json") as f:
        history = json.load(f)
        x_range = history["x_range"]
        iterations = history["iterations"]
        hid_lay = history["hid_lay"]
        num_equations = history["num_equations"]
        num_heads = history["num_heads"]

        loss_hist = history["loss_hist"]
        loss_hist["head"] = np.array(loss_hist["head"])

        alpha_list = history["alpha_list"]
        A_list = [torch.from_numpy(np.array(A)).to(device).double() for A in history["A"]]
        v_list = [torch.from_numpy(np.array(v)).to(device).double() for v in history["v"]]
        if history["force"] is None:
            force = None
            print("Force change with time")
        else:
            force = torch.from_numpy(np.array(history["force"])).to(device).double()
        if prev:
            trained_model = BuildNetwork_previous(1, hid_lay[0], hid_lay[1], hid_lay[2], num_equations, num_heads).to(device, dtype=torch.double)
        else:
            trained_model = BuildNetworkNew(1, hid_lay, num_equations, num_heads, device, activation="silu", IC_list=v_list).to(device, dtype=torch.double)
        trained_model.load_state_dict(torch.load(path_equation + str(model_file)))
        trained_model.eval()
    return trained_model, x_range, iterations, hid_lay, num_equations, num_heads, loss_hist, alpha_list, A_list, v_list, force
