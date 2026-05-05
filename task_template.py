import os
import sys
import torch
import pandas as pd
import requests
import random
import argparse
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.metrics import roc_curve
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.models import resnet18
import torchvision.transforms as transforms
import copy


# config
BASE = Path(__file__).parent
PUB_PATH = BASE / "pub.pt"
PRIV_PATH = BASE / "priv.pt"
MODEL_PATH = BASE / "model.pt"
OUTPUT_CSV = BASE / "submission.csv"

BASE_URL = "http://34.63.153.158"   #DONOT CHANGE
API_KEY = "011cad4716a5e5788dc998fd8fdabc6c"
TASK_ID = "01-mia"  #DONOT CHANGE



# dataset classes
class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform

    def __getitem__(self, index):
        id_ = self.ids[index]
        img = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)


class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []

    def __getitem__(self, index):
        id_, img, label = super().__getitem__(index)
        membership = self.membership[index] if self.membership[index] is not None else -1
        return id_, img, label, membership


# load datasets
print("Loading datasets...")
pub_ds = torch.load(PUB_PATH, weights_only=False)
priv_ds = torch.load(PRIV_PATH, weights_only=False)


MEAN = [0.7406, 0.5331, 0.7059]
STD  = [0.1491, 0.1864, 0.1301]

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.Normalize(mean=MEAN, std=STD),
])

pub_ds.transform  = transform
priv_ds.transform = transform

# ── model factory ────────────────────────────────────────────────────────────
# Architecture matches exactly what is specified in the assignment PDF:
# ResNet-18, modified for the dataset (small images, 9 classes)
def make_model():
    model = resnet18(weights=None)
    model.conv1   = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = torch.nn.Identity()
    model.fc      = torch.nn.Linear(512, 9)
    return model

# ── load target model ────────────────────────────────────────────────────────
print("Loading target model...")
target_model = make_model()
target_model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_model.eval().to(device)

# ── build reference model training set (non-members from pub_ds) ─────────────
# pub_ds has ground-truth membership labels — we train the reference model
# exclusively on NON-members so it has never seen any training data.
# LRA score = loss_ref(x) - loss_target(x)
# Members   → target loss LOW,  ref loss HIGH  → HIGH score  ✓
# Non-members → both losses similar             → score ~0

print("Identifying non-members in public set...")
nonmember_idx = [
    i for i in range(len(pub_ds))
    if pub_ds[i][3] == 0          # index 3 = membership
]
print(f"  Non-members available for reference training: {len(nonmember_idx)}")

ref_train_ds = Subset(pub_ds, nonmember_idx)
ref_loader   = DataLoader(ref_train_ds, batch_size=128, shuffle=True)

# ── train reference model ────────────────────────────────────────────────────
print("Training reference model on non-members...")
ref_model = make_model().to(device)
optimizer  = torch.optim.SGD(
    ref_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4
)
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

for epoch in range(30):
    ref_model.train()
    total_loss = 0.0
    for batch in ref_loader:
        ids, imgs, labels = batch[0], batch[1], batch[2]
        imgs   = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(ref_model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    if (epoch + 1) % 5 == 0:
        avg = total_loss / len(ref_loader)
        print(f"  Epoch {epoch+1:2d}/30 — avg loss: {avg:.4f}")

ref_model.eval()

# ── LRA score computation ────────────────────────────────────────────────────
def compute_lra_scores(dataset):
    loader     = DataLoader(dataset, batch_size=64, shuffle=False)
    all_ids    = []
    all_scores = []

    with torch.no_grad():
        for batch in loader:
            ids, imgs, labels = batch[0], batch[1], batch[2]
            imgs   = imgs.to(device)
            labels = labels.to(device)

            loss_target = F.cross_entropy(
                target_model(imgs), labels, reduction="none"
            )
            loss_ref = F.cross_entropy(
                ref_model(imgs), labels, reduction="none"
            )

            # positive score → likely member
            scores = loss_ref - loss_target

            all_ids.extend(ids)
            all_scores.extend(scores.cpu().numpy())

    scores_np = np.array(all_scores)
    scores_np = (scores_np - scores_np.min()) / (
        scores_np.max() - scores_np.min() + 1e-8
    )
    return all_ids, scores_np

# ── validate on public set ───────────────────────────────────────────────────
print("Validating LRA on public set...")
pub_ids, pub_scores = compute_lra_scores(pub_ds)
pub_labels = np.array(pub_ds.membership)

fpr, tpr, _ = roc_curve(pub_labels, pub_scores)
tpr_at_5fpr = tpr[fpr <= 0.05][-1] if np.any(fpr <= 0.05) else 0.0
print(f"TPR@5%FPR on public set: {tpr_at_5fpr:.4f}")

# ── score private set & save ─────────────────────────────────────────────────
print("Scoring private dataset...")
priv_ids, priv_scores = compute_lra_scores(priv_ds)

df = pd.DataFrame({
    "id":    [str(int(i)) for i in priv_ids],
    "score": priv_scores,
})
df.to_csv(OUTPUT_CSV, index=False)
print("Saved:", OUTPUT_CSV)





# submit
def die(msg):
    print(msg, file=sys.stderr)
    sys.exit(1)

parser = argparse.ArgumentParser(description="Submit a CSV file to the server.")
args = parser.parse_args()

submit_path = OUTPUT_CSV

if not submit_path.exists():
    die(f"File not found: {submit_path}")

try:
    with open(submit_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/submit/{TASK_ID}",
            headers={"X-API-Key": API_KEY},
            files={"file": (submit_path.name, f, "application/csv")},
            timeout=(10, 600),
        )
    try:
        body = resp.json()
    except Exception:
        body = {"raw_text": resp.text}

    if resp.status_code == 413:
        die("Upload rejected: file too large (HTTP 413).")

    resp.raise_for_status()

    print("Successfully submitted.")
    print("Server response:", body)
    submission_id = body.get("submission_id")
    if submission_id:
        print(f"Submission ID: {submission_id}")

except requests.exceptions.RequestException as e:
    detail = getattr(e, "response", None)
    print(f"Submission error: {e}")
    if detail is not None:
        try:
            print("Server response:", detail.json())
        except Exception:
            print("Server response (text):", detail.text)
    sys.exit(1)

