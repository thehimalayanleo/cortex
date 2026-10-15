"""PRUNE-1 diagnostic (post hoc, reported only): is the corrected target implemented right, and where does it hurt?

CPU only (the GPU is busy). For the first LAYERS decoder layers of the student, with a Wanda mask at 50%:
  1. per linear: fit error to the dense output ||Xp w' - Xd w||^2 for the standard refit and the corrected refit.
     The corrected refit minimizes exactly this, so it must be lower. If not, the plumbing is wrong.
  2. per layer: error of the whole layer output (residual stream) against the dense model's layer output, after
     pruning all linears of the layer with each refit, inputs carried forward on each path separately.
    python prune1_diag.py STUDENT [LAYERS] [NCAL]
"""
import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prune1  # noqa: E402

prune1.dev = "cpu"
torch.set_num_threads(16)


def refit(Wt, Hd, B, keep):
    out = torch.zeros_like(Wt)
    for i in range(Wt.shape[0]):
        S = keep[i].nonzero().squeeze(1)
        out[i, S] = torch.linalg.solve(Hd[S][:, S], B[i, S])
    return out


@torch.no_grad()
def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    student = sys.argv[1]; LAYERS = int(sys.argv[2]) if len(sys.argv) > 2 else 4; NCAL = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    tok = AutoTokenizer.from_pretrained(student)
    model = AutoModelForCausalLM.from_pretrained(student, dtype=torch.float32).eval()
    texts = prune1.calib_texts(tok, 0, NCAL)
    layers = model.model.layers; caught = []

    class Catch(nn.Module):
        def __init__(s, m):
            super().__init__(); s.m = m

        def forward(s, *a, **kw):
            x = a[0] if a else kw.pop("hidden_states"); caught.append((x, kw)); raise ValueError
    layers[0] = Catch(layers[0])
    for t in texts:
        try:
            model(tok(t, return_tensors="pt", add_special_tokens=False).input_ids[:, :1024])
        except ValueError:
            pass
    layers[0] = layers[0].m
    kws = [k for _, k in caught]
    paths = {name: [x.clone() for x, _ in caught] for name in ("std", "cor")}
    xd = [x.clone() for x, _ in caught]
    report = []
    for li in range(LAYERS):
        dense = layers[li]
        out_d = [(dense(xd[j], **kws[j])) for j in range(len(xd))]
        out_d = [o[0] if isinstance(o, tuple) else o for o in out_d]
        row = {"layer": li}
        for name in ("std", "cor"):
            layer = copy.deepcopy(dense)
            lins = {n: m for n, m in layer.named_modules() if isinstance(m, nn.Linear)}
            dl = {n: m for n, m in dense.named_modules() if isinstance(m, nn.Linear)}
            H = {n: 0 for n in lins}; C = {n: 0 for n in lins}; xn = {n: 0 for n in lins}; Xs = {n: [] for n in lins}
            for j in range(len(xd)):
                gp, gd = {}, {}
                hp = [m.register_forward_hook(lambda mod, i, o, n=n: gp.__setitem__(n, i[0].reshape(-1, i[0].shape[-1]))) for n, m in lins.items()]
                layer(paths[name][j], **kws[j]); [h.remove() for h in hp]
                hd = [m.register_forward_hook(lambda mod, i, o, n=n: gd.__setitem__(n, i[0].reshape(-1, i[0].shape[-1]))) for n, m in dl.items()]
                dense(xd[j], **kws[j]); [h.remove() for h in hd]
                for n in lins:
                    H[n] = H[n] + gp[n].T @ gp[n]; C[n] = C[n] + gp[n].T @ gd[n]; xn[n] = xn[n] + (gp[n] ** 2).sum(0)
                    if j < 4:
                        Xs[n].append((gp[n], gd[n]))
            fit = {}
            for n, m in lins.items():
                Wt = m.weight.data.clone(); d = Wt.shape[1]
                damp = 0.01 * torch.mean(torch.diag(H[n])); I = torch.eye(d)
                keep = prune1.wanda_keep(Wt, xn[n].sqrt(), 0.5)
                B = Wt @ (H[n] + damp * I) if name == "std" else Wt @ (C[n].T + damp * I)
                new = refit(Wt, H[n] + damp * I, B, keep)
                err = sum(float(((xp @ new.T - xdd @ Wt.T) ** 2).sum()) for xp, xdd in Xs[n])
                ref = sum(float(((xdd @ Wt.T) ** 2).sum()) for _, xdd in Xs[n])
                fit[n] = round(err / ref, 5)
                m.weight.data = new
            out_p = []
            for j in range(len(xd)):
                o = layer(paths[name][j], **kws[j]); out_p.append(o[0] if isinstance(o, tuple) else o)
            layer_err = sum(float(((a - b) ** 2).sum()) for a, b in zip(out_p, out_d)) / sum(float((b ** 2).sum()) for b in out_d)
            row[name] = {"linear_fit_err_to_dense": fit, "layer_out_err_to_dense": round(layer_err, 6)}
            paths[name] = out_p
        xd = out_d
        print(json.dumps(row), flush=True)
        report.append(row)
    Path(Path.home() / "prune1" / "diag.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
