from utils import *
def build_dpo_dataset(good_ratio = None):
    y1 = torch.normal(REF_MU, REF_SIGMA, size=(DATASET_SIZE,))
    y2 = torch.normal(REF_MU, REF_SIGMA, size=(DATASET_SIZE,))

    dist1 = torch.abs(y1 - TARGET)
    dist2 = torch.abs(y2 - TARGET)

    winners = torch.where(dist1 < dist2, y1, y2)
    losers  = torch.where(dist1 < dist2, y2, y1)
    if good_ratio:
        margin = torch.abs(dist1 - dist2)

        k = int(good_ratio * DATASET_SIZE)
        topk_indices = torch.topk(margin, k=k).indices

        return winners[topk_indices], losers[topk_indices]
    return winners, losers


def build_kto_dataset(delta, good_ratio = None):
    zone = (ZONE[0] - delta, ZONE[1] + delta)
    y = torch.normal(REF_MU, REF_SIGMA, size=(DATASET_SIZE,))
    labels = ((y >= zone[0]) & (y <= zone[1])).float()
    good = y[labels == 1]
    bad = y[labels == 0]
    if good_ratio:
        n_good = int(DATASET_SIZE * good_ratio)
        n_bad = DATASET_SIZE - n_good

        good_samples = []
        bad_samples = []

        while len(good_samples) < n_good or len(bad_samples) < n_bad:
            y_batch = torch.normal(REF_MU, REF_SIGMA, size=(DATASET_SIZE,))
            labels_batch = ((y_batch >= zone[0]) & (y_batch <= zone[1]))

            good_batch = y_batch[labels_batch]
            bad_batch = y_batch[~labels_batch]

            good_samples.extend(good_batch.tolist())
            bad_samples.extend(bad_batch.tolist())

        good_tensor = torch.tensor(good_samples[:n_good])
        bad_tensor = torch.tensor(bad_samples[:n_bad])

        y = torch.cat([good_tensor, bad_tensor])
        labels = torch.cat([
            torch.ones(n_good),
            torch.zeros(n_bad)
        ])
    perm = torch.randperm(DATASET_SIZE)
    y = y[perm]
    labels = labels[perm]
    return y, labels