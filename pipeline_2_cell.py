# ============================================================================
# PIPELINE 2 : RL PREPROCESSING + AGGREGATION STRATEGY COMPARISON (no privacy)
# One cell. Set USE_RL at the top:
#   USE_RL = False -> control run (fixed preprocessing, no enhancement)
#   USE_RL = True  -> RL agent picks the enhancement each round
# 5 models x 3 strategies (FedAvg, FedProx, FedMomentum), 20 rounds each.
# FedMomentum uses the corrected settings (beta 0.5, server LR 0.7,
# BatchNorm running stats excluded from momentum).
# ============================================================================
import copy, json, warnings
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision
warnings.filterwarnings("ignore")

# ---------------- SWITCH ----------------
USE_RL = False                                   # <-- change to True for the RL run
TAG = "with_rl" if USE_RL else "no_rl"
SAVE_PATH = f"/kaggle/working/pipeline2_{TAG}.json"

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE, "| GPUs:", torch.cuda.device_count(), "| RL:", USE_RL)

# ---------------- CONFIG ----------------
ROUNDS = 20; LOCAL_EPOCHS = 2; CLIENT_LR = 0.05; BATCH_SIZE = 64; EVAL_EVERY = 5
SAMPLE_CLIENTS = 15
P2_MU = 0.01                                     # FedProx
P2_BETA = 0.5; SERVER_LR = 0.7                   # FedMomentum (corrected)
STRATEGIES = ["FedAvg", "FedProx", "FedMomentum"]
MODELS_TO_RUN = ["CNN_Small", "CNN_MobileNetV2", "CNN_ShuffleNet",
                 "Transformer_DeiT", "Fusion_MNetV2_DeiT"]
DIRICHLET_A = 0.5; NUM_CLIENTS = 15
TRAIN_CSV = "/kaggle/input/datasets/crawford/emnist/emnist-balanced-train.csv"
TEST_CSV = "/kaggle/input/datasets/crawford/emnist/emnist-balanced-test.csv"

# ---------------- DATA (EMNIST CSV + Dirichlet non-IID split) ----------------
def load_data():
    def read(p):
        df = pd.read_csv(p, header=None); y = df.iloc[:, 0].to_numpy().astype(np.int64)
        x = df.iloc[:, 1:].to_numpy().astype(np.float32).reshape(-1, 28, 28)
        x = np.transpose(x, (0, 2, 1)) / 255.0          # EMNIST is stored transposed
        return torch.tensor(x[:, None]), torch.tensor(y)
    xtr, ytr = read(TRAIN_CSV); xte, yte = read(TEST_CSV)
    rng = np.random.default_rng(SEED); lab = ytr.numpy(); NC = 47
    idxc = [np.where(lab == c)[0] for c in range(NC)]; parts = [[] for _ in range(NUM_CLIENTS)]
    for c in range(NC):
        rng.shuffle(idxc[c]); pr = rng.dirichlet([DIRICHLET_A] * NUM_CLIENTS)
        cuts = (np.cumsum(pr) * len(idxc[c])).astype(int)[:-1]
        for i, pt in enumerate(np.split(idxc[c], cuts)): parts[i].extend(pt.tolist())
    return [(xtr[np.array(p)], ytr[np.array(p)]) for p in parts], xte, yte, NC

def make_loader(x, y, bs=BATCH_SIZE, shuffle=True):
    return DataLoader(TensorDataset(x, y), batch_size=bs, shuffle=shuffle, num_workers=0)

# ---------------- MODELS ----------------
def build_models(nc):
    class CustomCNN(nn.Module):
        def __init__(s):
            super().__init__()
            s.f = nn.Sequential(nn.Conv2d(1,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU())
            s.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, nc))
        def forward(s, x): return s.head(s.f(x))
    def mobilenetv2():
        m = torchvision.models.mobilenet_v2(weights=None)
        m.features[0][0] = nn.Conv2d(1, 32, 3, 1, 1, bias=False); m.classifier[1] = nn.Linear(m.last_channel, nc); return m
    def shufflenet():
        m = torchvision.models.shufflenet_v2_x0_5(weights=None)
        m.conv1[0] = nn.Conv2d(1, 24, 3, 1, 1, bias=False); m.maxpool = nn.Identity()
        m.fc = nn.Linear(m.fc.in_features, nc); return m
    class TokenTransformer(nn.Module):
        def __init__(s, embed, depth, heads, nt):
            super().__init__()
            s.cls = nn.Parameter(torch.zeros(1,1,embed)); s.pos = nn.Parameter(torch.zeros(1,nt+1,embed))
            l = nn.TransformerEncoderLayer(embed, heads, embed*2, 0.1, batch_first=True, activation="gelu")
            s.enc = nn.TransformerEncoder(l, depth); s.norm = nn.LayerNorm(embed); s.head = nn.Linear(embed, nc)
        def forward(s, t):
            B = t.size(0); z = torch.cat([s.cls.expand(B,-1,-1), t], 1) + s.pos
            return s.head(s.norm(s.enc(z)[:, 0]))
    class DeiTSmall(nn.Module):
        def __init__(s, embed=128, depth=4, heads=4, patch=4):
            super().__init__(); nt = (28 // patch) ** 2
            s.embed = nn.Conv2d(1, embed, patch, patch); s.tr = TokenTransformer(embed, depth, heads, nt)
        def forward(s, x): return s.tr(s.embed(x).flatten(2).transpose(1, 2))
    class Fusion(nn.Module):
        def __init__(s, embed=128, grid=4):
            super().__init__(); mb = torchvision.models.mobilenet_v2(weights=None)
            mb.features[0][0] = nn.Conv2d(1, 32, 3, 1, 1, bias=False)
            s.cnn = mb.features; s.proj = nn.Conv2d(mb.last_channel, embed, 1); s.grid = grid
            s.tr = TokenTransformer(embed, 4, 4, grid * grid)
        def forward(s, x):
            f = F.adaptive_avg_pool2d(s.cnn(x), (s.grid, s.grid))
            return s.tr(s.proj(f).flatten(2).transpose(1, 2))
    return {"CNN_Small": lambda: CustomCNN(), "CNN_MobileNetV2": lambda: mobilenetv2(),
            "CNN_ShuffleNet": lambda: shufflenet(), "Transformer_DeiT": lambda: DeiTSmall(),
            "Fusion_MNetV2_DeiT": lambda: Fusion()}

# ---------------- FL helpers ----------------
def get_params(m): return {k: v.detach().clone() for k, v in m.state_dict().items()}
def set_params(m, p): m.load_state_dict(p, strict=True)

@torch.no_grad()
def evaluate(m, loader):
    m.eval(); c = t = 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE); c += (m(xb).argmax(1) == yb).sum().item(); t += yb.size(0)
    return 100.0 * c / max(t, 1)

def local_train(model, loader, epochs, lr, gp=None, mu=0.0):
    """Local SGD. mu > 0 adds the FedProx proximal term."""
    model.train(); opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9); ce = nn.CrossEntropyLoss()
    g = [gp[k].to(DEVICE) for k, _ in model.named_parameters()] if (mu > 0 and gp) else None
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE); opt.zero_grad(); loss = ce(model(xb), yb)
            if g is not None:
                loss = loss + (mu / 2) * sum(((p - q) ** 2).sum() for p, q in zip(model.parameters(), g))
            loss.backward(); opt.step()
    return get_params(model)

def weighted_average(states, weights):
    w = np.array(weights, float); w = w / w.sum(); avg = copy.deepcopy(states[0])
    for k in avg:
        if avg[k].dtype.is_floating_point:
            avg[k] = torch.stack([s[k].float() * float(wi) for s, wi in zip(states, w)], 0).sum(0).to(avg[k].dtype)
        else: avg[k] = states[0][k]
    return avg

def fedmomentum(gs, new_avg, vel, beta):
    """Server momentum. BatchNorm running stats are excluded (they diverged at beta 0.9)."""
    out = copy.deepcopy(new_avg)
    keys = [k for k, v in gs.items() if v.dtype.is_floating_point and "running_" not in k]
    if vel is None: vel = {k: torch.zeros_like(gs[k].float()) for k in keys}
    for k in keys:
        d = new_avg[k].float() - gs[k].float(); vel[k] = beta * vel[k] + d
        out[k] = (gs[k].float() + SERVER_LR * vel[k]).to(gs[k].dtype)
    return out, vel

# ---------------- RL agent: epsilon-greedy multi-armed bandit ----------------
# Reward = global-model accuracy on one enhanced batch.  Q[a] += (reward - Q[a]) / N[a]
class RLPreprocessor:
    def __init__(s, eps=0.2):
        s.actions = ["identity", "sharpen", "gamma", "contrast"]
        s.q = np.zeros(4); s.n = np.zeros(4); s.eps = eps
        s._sh = torch.tensor([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=torch.float32).view(1,1,3,3)
    def _apply(s, x, a):
        if a == "identity": return x
        if a == "sharpen": return F.conv2d(x, s._sh.to(x.device), padding=1).clamp(0, 1)
        if a == "gamma": return x.clamp(0, 1) ** 0.7
        if a == "contrast": return ((x - 0.5) * 1.5 + 0.5).clamp(0, 1)
        return x
    def select(s): return np.random.randint(4) if np.random.rand() < s.eps else int(np.argmax(s.q))
    def reward(s, a, model, xb, yb):
        model.eval()
        with torch.no_grad():
            acc = (model(s._apply(xb.to(DEVICE), s.actions[a])).argmax(1) == yb.to(DEVICE)).float().mean().item()
        s.n[a] += 1; s.q[a] += (acc - s.q[a]) / s.n[a]; return acc
    def transform(s, x, a): return s._apply(x.to(DEVICE), s.actions[a]).cpu()

# ---------------- federated loop ----------------
def run(model_name, strategy, clients, test_loader, builders):
    print(f"\n{'='*58}\n{TAG} | {model_name} | {strategy}\n{'='*58}")
    model = builders[model_name]().to(DEVICE); gs = get_params(model)
    rl = RLPreprocessor() if USE_RL else None
    vel = None; mu = P2_MU if strategy == "FedProx" else 0.0
    n_s = min(SAMPLE_CLIENTS, len(clients)); history = []; actions_log = []
    for rnd in range(1, ROUNDS + 1):
        chosen = np.random.choice(len(clients), n_s, replace=False); states = []; wts = []
        action = rl.select() if USE_RL else None
        if USE_RL: actions_log.append(rl.actions[action])
        for cid in chosen:
            cx, cy = clients[cid]; set_params(model, gs)
            if USE_RL:
                rl.reward(action, model, cx[:BATCH_SIZE], cy[:BATCH_SIZE]); data = rl.transform(cx, action)
            else:
                data = cx                                         # fixed preprocessing
            states.append(local_train(model, make_loader(data, cy), LOCAL_EPOCHS, CLIENT_LR, gs, mu)); wts.append(len(cy))
        avg = weighted_average(states, wts)
        gs, vel = (fedmomentum(gs, avg, vel, P2_BETA) if strategy == "FedMomentum" else (avg, vel))
        if rnd % EVAL_EVERY == 0 or rnd == ROUNDS:
            set_params(model, gs); acc = evaluate(model, test_loader); history.append((rnd, acc))
            print(f"  round {rnd:>3} | acc {acc:6.2f}% | action -> {rl.actions[action] if USE_RL else 'identity'}")
    set_params(model, gs); acc = evaluate(model, test_loader)
    print(f"  FINAL | {strategy} acc {acc:.2f}%")
    return {"acc": acc, "history": history,
            "rl_q": dict(zip(rl.actions, rl.q.tolist())) if USE_RL else {},
            "rl_actions_per_round": actions_log}

# ---------------- MAIN ----------------
if __name__ == "__main__":
    clients, test_x, test_y, NUM_CLASSES = load_data()
    print(f"clients={len(clients)} | classes={NUM_CLASSES} | test={test_x.shape[0]}")
    builders = build_models(NUM_CLASSES); test_loader = make_loader(test_x, test_y, shuffle=False)
    RESULTS = {}
    for name in MODELS_TO_RUN:
        for strat in STRATEGIES:
            RESULTS[f"{name}|{strat}"] = run(name, strat, clients, test_loader, builders)
            json.dump(RESULTS, open(SAVE_PATH, "w"), indent=2)      # crash-safe save

    print("\n" + "=" * 60); print(f"PIPELINE 2 RESULTS ({TAG})"); print("=" * 60)
    print(f"{'Model':<22}{'FedAvg':<11}{'FedProx':<11}{'FedMomentum':<13}"); print("-" * 60)
    for name in MODELS_TO_RUN:
        a = [RESULTS[f"{name}|{s}"]["acc"] for s in STRATEGIES]
        print(f"{name:<22}{a[0]:<11.2f}{a[1]:<11.2f}{a[2]:<13.2f}")
    print("=" * 60); print(f"[saved to {SAVE_PATH}]")
