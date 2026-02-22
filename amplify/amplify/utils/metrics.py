import torch
from einops import repeat


def compute_cross_track_distance(ref_traj, sampled_trajs, device, discount=None):
    """
    Calculate the (normalized) distance to closest points in reference
    args:
        ref_traj: (bs, v, horizon, n_tracks, 2)
        sampled_trajs: (bs, v, horizon, n_tracks, 2)
        discount: discount factor for each timestep
    returns:
        cost: (n_samples,)
    """
    bs, v, h, nt, _ = sampled_trajs.shape # (batch size, horizon, num_tracks, 2)

    # getting all combinations of distances between tracks (bs, horizon, n_tracks, n_tracks, 2)
    samples_expanded = sampled_trajs.unsqueeze(2) # (bs, v, 1, horizon, n_tracks, 2)
    ref_traj_expanded = ref_traj.unsqueeze(3) # (bs, v, horizon, 1, n_tracks, 2)
    pairwise_distances = torch.norm(samples_expanded - ref_traj_expanded, dim=-1) # (bs, v, horizon, horizon, n_tracks)

    # find the minimum distance between tracks
    min_distances = torch.min(pairwise_distances, dim=3).values # (bs, v, horizon, n_tracks)

    if discount is not None:
        discounts = torch.tensor([discount ** t for t in range(h)], device=device)
        discounts = repeat(discounts, "h -> bs v h nt", bs=bs, v=v, nt=nt)
        traj_distances = torch.mean(min_distances * discounts, dim=(1, 2, 3)) # average over horizon and num_queries to get (bs,)
    else:
        traj_distances = min_distances.mean(dim=(1, 2, 3)) # average over horizon and num_queries without discounting

    return traj_distances.mean() # mean over batch


def get_traj_metrics(pred_traj, gt_traj, img_size, all_pixel_tol=False, motion_tokenizer=None, histogram=False):
    """
    Args:
        pred (torch.Tensor): (bs, num_views, seq_len, num_queries, 2)
        gt (torch.Tensor): (bs, num_views, seq_len, num_queries, 2)
        img_size (tuple): (height, width)
    """
    metrics = {}
    pred_traj_velocities = pred_traj[:, :, 1:] - pred_traj[:, :, :-1]
    gt_traj_velocities = gt_traj[:, :, 1:] - gt_traj[:, :, :-1]

    # mse
    metrics['mse'] = torch.nn.functional.mse_loss(pred_traj, gt_traj)

    # l1
    metrics['l1'] = torch.nn.functional.l1_loss(pred_traj, gt_traj)

    # cross track distance (l2 distance between the predicted and closest point on target trajectory)
    metrics['cross_track_l2'] = compute_cross_track_distance(ref_traj=pred_traj, sampled_trajs=gt_traj, device=pred_traj.device)

    if histogram: # histogram of normalized l2 distance errors
        errors = torch.norm(pred_traj - gt_traj, dim=-1)
        errors = errors.flatten()
        norm_abs_err_histogram_np = np.histogram(errors.detach().cpu().numpy(), bins=500, range=(0, 1))
        metrics['norm_abs_err_histogram_np'] = norm_abs_err_histogram_np

    norm_pixel_size = 2.0 / img_size[0] # x2 because the normalized coords are in (-1,1). This also assumes square image

    metrics['normalized_accuracy'] = (torch.sum(torch.abs(pred_traj - gt_traj) < (norm_pixel_size / 2)) / gt_traj.numel()) # assumes square image

    # Delta AUC (average accuracy over several tolerances)
    accuracies = []
    for tolerance in torch.arange(1, 11, 1): # 1 to 10
        accuracy = torch.sum(torch.abs(pred_traj - gt_traj) < (tolerance * norm_pixel_size)) / gt_traj.numel()
        if all_pixel_tol:
            metrics[f'pixel_accuracy_{tolerance}'] = accuracy
        accuracies.append(accuracy.item())
    metrics['delta_auc'] = torch.tensor(accuracies).mean()

    if motion_tokenizer is not None:
        metrics['codebook_accuracy'] = get_codebook_accuracy(motion_tokenizer, pred_traj_velocities, gt_traj_velocities)

    metrics['nonzero_pred_percent'], metrics['nonzero_gt_percent'] = get_nonzero_pred_percent(pred_traj_velocities, gt_traj_velocities)
    metrics['nonzero_pred_accuracy'] = get_nonzero_pred_accuracy(pred_traj_velocities, gt_traj_velocities)
    metrics['nonzero_pred_f1'] = get_nonzero_pred_f1(pred_traj_velocities, gt_traj_velocities)

    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            metrics[k] = v.detach().item()

    return metrics


def get_codebook_perplexity(indices, codebook_size):
    counts = torch.bincount(indices.flatten(), minlength=codebook_size).float()
    usage_frequency = counts / counts.sum()  # Relative frequency of each code
    perplexity = torch.pow(2, -torch.sum(usage_frequency * torch.log2(usage_frequency + 1e-10)))

    return perplexity


def get_normalized_codebook_perplexity(indices, codebook_size):
    counts = torch.bincount(indices.flatten(), minlength=codebook_size).float()
    usage_frequency = counts / counts.sum()  # Relative frequency of each code
    perplexity = torch.pow(2, -torch.sum(usage_frequency * torch.log2(usage_frequency + 1e-10)))

    # Normalize by the maximum possible perplexity given num_indices and codebook_size
    num_indices = indices.numel()  # Total number of codes used in this batch
    max_perplexity = min(codebook_size, num_indices)
    normalized_perplexity = perplexity / max_perplexity

    return normalized_perplexity


def get_codebook_accuracy(motion_tokenizer, pred_traj_velocities, gt_traj_velocities):
    """
    args:
        motion_tokenizer: MotionTokenizer
        pred_traj: (b, v, t, n, d)
        gt_traj: (b, v, t, n, d)
    """
    _, pred_codebook_indices, _ = motion_tokenizer(pred_traj_velocities)
    _, gt_codebook_indices, _ = motion_tokenizer(gt_traj_velocities)
    return (pred_codebook_indices == gt_codebook_indices).sum() / (pred_codebook_indices.shape[0]*pred_codebook_indices.shape[1])


def get_nonzero_pred_percent(pred_traj_velocities, gt_traj_velocities):
    """
    Computes the fraction of nonzero elements for predictions and ground truth.

    Args:
        pred_traj_velocities: (b, v, t, n, d) tensor.
        gt_traj_velocities: (b, v, t, n, d) tensor.

    Returns:
        (pred_nonzero, gt_nonzero): Tuple of fractions (in [0, 1]).
    """
    pred_traj_nonzero = (pred_traj_velocities != 0).sum() / pred_traj_velocities.numel()
    gt_traj_nonzero = (gt_traj_velocities != 0).sum() / gt_traj_velocities.numel()
    return pred_traj_nonzero, gt_traj_nonzero


def get_true_false_positive_negative(pred_traj_velocities, gt_traj_velocities):
    """
    Computes boolean masks for true positives, false positives, true negatives, and false negatives.

    Args:
        pred_traj_velocities: (b, v, t, n, d) tensor.
        gt_traj_velocities: (b, v, t, n, d) tensor.

    Returns:
        Tuple of boolean tensors: (tp, fp, tn, fn).
    """
    tp = (pred_traj_velocities != 0) & (gt_traj_velocities != 0)
    fp = (pred_traj_velocities != 0) & (gt_traj_velocities == 0)
    tn = (pred_traj_velocities == 0) & (gt_traj_velocities == 0)
    fn = (pred_traj_velocities == 0) & (gt_traj_velocities != 0)
    return tp, fp, tn, fn


def get_nonzero_pred_accuracy(pred_traj_velocities, gt_traj_velocities):
    """
    Computes the overall accuracy for zero/nonzero predictions.

    Args:
        pred_traj_velocities: (b, v, t, n, d) tensor.
        gt_traj_velocities: (b, v, t, n, d) tensor.

    Returns:
        Accuracy: Fraction of correctly predicted zeros and nonzeros.
    """
    tp, fp, tn, fn = get_true_false_positive_negative(pred_traj_velocities, gt_traj_velocities)
    correct = tp.sum() + tn.sum()
    total = tp.sum() + fp.sum() + tn.sum() + fn.sum()
    return correct / total if total != 0 else 0.0


def get_tfpn_percent(pred_traj_velocities, gt_traj_velocities):
    """
    Computes the proportions of true/false positives and negatives within their respective groups.

    Args:
        pred_traj_velocities: (b, v, t, n, d) tensor.
        gt_traj_velocities: (b, v, t, n, d) tensor.

    Returns:
        Tuple: (tp_percent, fp_percent, tn_percent, fn_percent)
        where tp_percent and fp_percent are computed over (tp + fp),
        and tn_percent and fn_percent are computed over (tn + fn).
    """
    tp, fp, tn, fn = get_true_false_positive_negative(pred_traj_velocities, gt_traj_velocities)
    pos_denom = tp.sum() + fp.sum()
    neg_denom = tn.sum() + fn.sum()

    tp_percent = tp.sum() / pos_denom if pos_denom != 0 else 0.0
    fp_percent = fp.sum() / pos_denom if pos_denom != 0 else 0.0
    tn_percent = tn.sum() / neg_denom if neg_denom != 0 else 0.0
    fn_percent = fn.sum() / neg_denom if neg_denom != 0 else 0.0

    return tp_percent, fp_percent, tn_percent, fn_percent


def get_nonzero_pred_f1(pred_traj_velocities, gt_traj_velocities):
    """
    Computes the F1 score for nonzero predictions.

    Args:
        pred_traj_velocities: (b, v, t, n, d) tensor.
        gt_traj_velocities: (b, v, t, n, d) tensor.

    Returns:
        F1 score computed as:
            F1 = tp / (tp + 0.5 * (fp + fn))
    """
    tp, fp, tn, fn = get_true_false_positive_negative(pred_traj_velocities, gt_traj_velocities)
    tp_sum = tp.sum()
    fp_sum = fp.sum()
    fn_sum = fn.sum()

    denominator = tp_sum + 0.5 * (fp_sum + fn_sum)
    return tp_sum / denominator if denominator != 0 else 0.0
