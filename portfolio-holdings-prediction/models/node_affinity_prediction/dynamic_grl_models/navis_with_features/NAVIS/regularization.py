import torch
import torch.nn as nn
import torch.nn.functional as F

class HindgeRegularization(nn.Module):
    """
    RankNet-based optimization criterion for ranking classes.

    This criterion implements the loss function described in the paper:
    L = L1 + L2

    Where:
    - L1: Loss for pairs where both classes are in true top classes
    - L2: Loss for pairs where one class is in true top classes and one is in predicted top classes

    The pairwise loss function is: l(a,b) = max(0, δ - a-b)
    Fully vectorized implementation for GPU acceleration.
    """

    def __init__(self, top_k=10, delta=0.01):
        """
        Args:
            top_k (int): Number of top classes to consider (default: 10)
            eps (float): Small constant for numerical stability (default: 1e-8)
        """
        super(HindgeRegularization, self).__init__()
        self.top_k = top_k
        self.delta = delta

    def forward(self, scores, true_labels):
        """
        Forward pass of the criterion.

        Args:
            scores (torch.Tensor): Predicted scores for all classes, shape (batch_size, num_classes)
            true_labels (torch.Tensor): Ground truth labels, shape (batch_size, num_classes)
                                      Can be binary (0/1) or contain ranking information

        Returns:
            torch.Tensor: Computed loss value
        """
        batch_size, num_classes = scores.shape
        device = scores.device

        # Get top-k predicted classes for all batches
        _, pred_top_indices = torch.topk(scores, self.top_k, dim=1, largest=True)  # (batch_size, top_k)

        # Get top-k true classes for all batches
        if torch.all((true_labels == 0) | (true_labels == 1)):
            # Binary case: handle cases where we might have more or fewer than top_k true classes
            true_top_indices, true_valid_mask = self._get_binary_top_k(scores, true_labels)
            l3_loss = 0
        else:
            # Ranking case: select top-k based on label values, but ignore classes with label == 0
            true_mask = (true_labels > 0)  # Only consider classes with positive labels
            masked_labels = torch.where(true_mask, true_labels, torch.full_like(true_labels, -float('inf')))
            _, true_top_indices = torch.topk(masked_labels, self.top_k, dim=1, largest=True)

            # Create validity mask for ranking case
            batch_size = scores.shape[0]
            num_true_per_batch = true_mask.sum(dim=1)  # (batch_size,)
            position_indices = torch.arange(self.top_k, device=scores.device).unsqueeze(0)  # (1, top_k)
            true_valid_mask = position_indices < num_true_per_batch.unsqueeze(1)  # (batch_size, top_k)

            num_non_zero_labels = true_labels.sum(dim=1, keepdim=True)  # (batch_size, 1)

            # sum only scores with non-zero true labels
            non_zero_scores = torch.sum(scores * (true_labels != 0), dim=1, keepdim=True)

            mins = non_zero_scores / (num_non_zero_labels + 1e-8)

            non_zero_mask = true_labels > 0

            v = torch.maximum(torch.zeros_like(mins), - (mins - scores) + 2 * self.delta)
            v[non_zero_mask] = 0
            l3_loss = torch.sum(v, dim=1)
            l3_loss = l3_loss.mean()  # Average over batch

        # Compute L1 loss: pairwise comparisons within true top classes
        l1_loss = self._compute_l1_loss(scores, true_top_indices, true_valid_mask)

        # Compute L2 loss: true top vs predicted top (not in true top)
        l2_loss = self._compute_l2_loss(scores, pred_top_indices, true_top_indices, true_valid_mask)

        return l1_loss + l2_loss + l3_loss

    def _get_binary_top_k(self, scores, true_labels):
        """
        Get top-k true classes for binary labels case.
        Only considers classes with true_labels == 1, ignores classes with true_labels == 0.

        Args:
            scores: Predicted scores (batch_size, num_classes)
            true_labels: Binary labels (batch_size, num_classes)

        Returns:
            torch.Tensor: Top-k true class indices (batch_size, top_k)
            torch.Tensor: Valid mask indicating which entries in top_k are actual true classes
        """
        batch_size, num_classes = scores.shape

        # Create a mask for true classes (where label == 1)
        true_mask = (true_labels == 1)  # (batch_size, num_classes)

        # Count number of true classes per batch
        num_true_per_batch = true_mask.sum(dim=1)  # (batch_size,)

        # Set scores of non-true classes to very negative values
        masked_scores = torch.where(true_mask, scores, torch.full_like(scores, -float('inf')))

        # Get top-k from the masked scores
        _, true_top_indices = torch.topk(masked_scores, self.top_k, dim=1, largest=True)

        # Create validity mask - True for positions that correspond to actual true classes
        # (i.e., not padded with -inf values)
        batch_indices = torch.arange(batch_size, device=scores.device).unsqueeze(1)  # (batch_size, 1)
        position_indices = torch.arange(self.top_k, device=scores.device).unsqueeze(0)  # (1, top_k)
        valid_mask = position_indices < num_true_per_batch.unsqueeze(1)  # (batch_size, top_k)

        return true_top_indices, valid_mask

    def _compute_l1_loss(self, scores, true_top_indices, true_valid_mask):
        """
        Compute L1 loss: pairwise comparisons within true top classes.
        Only considers valid true classes (ignores padded entries from classes with label 0).

        Args:
            scores: Predicted scores (batch_size, num_classes)
            true_top_indices: True top class indices (batch_size, top_k)
            true_valid_mask: Mask indicating valid true classes (batch_size, top_k)

        Returns:
            torch.Tensor: L1 loss value
        """
        batch_size, top_k = true_top_indices.shape

        # Get scores for true top classes
        true_scores = scores.gather(1, true_top_indices)  # (batch_size, top_k)

        # Create all pairwise combinations
        # Expand dimensions for broadcasting
        scores_i = true_scores.unsqueeze(2)  # (batch_size, top_k, 1)
        scores_j = true_scores.unsqueeze(1)  # (batch_size, 1, top_k)

        # Compute pairwise score differences: S(ci) - S(cj)
        score_diffs = scores_i - scores_j  # (batch_size, top_k, top_k)

        # Create mask for valid pairs (i < j to avoid duplicates and self-pairs)
        i_indices = torch.arange(top_k, device=scores.device).unsqueeze(1)  # (top_k, 1)
        j_indices = torch.arange(top_k, device=scores.device).unsqueeze(0)  # (1, top_k)
        pair_mask = (i_indices < j_indices).unsqueeze(0)  # (1, top_k, top_k)

        # Create validity mask for pairs - both classes must be valid
        valid_i = true_valid_mask.unsqueeze(2)  # (batch_size, top_k, 1)
        valid_j = true_valid_mask.unsqueeze(1)  # (batch_size, 1, top_k)
        valid_pairs = valid_i & valid_j  # (batch_size, top_k, top_k)

        # Combine pair mask with validity mask
        final_mask = pair_mask.float() * valid_pairs.float()

        # For pairs where i < j, ci should be ranked higher than cj (based on position in true_top_indices)
        # So we want S(ci) > S(cj), meaning S(ci) - S(cj) should be positive
        pairwise_losses = torch.maximum(torch.zeros_like(score_diffs), -score_diffs + self.delta)

        # Apply mask and sum
        masked_losses = pairwise_losses * final_mask
        l1_loss = masked_losses.sum(dim=(1, 2))  # Sum over all pairs for each batch

        # Count valid pairs for proper averaging
        valid_pair_counts = final_mask.sum(dim=(1, 2))  # (batch_size,)
        valid_pair_counts = torch.clamp(valid_pair_counts, min=1)  # Avoid division by zero

        # Average by number of valid pairs per batch
        l1_loss = l1_loss / valid_pair_counts

        return l1_loss.mean()  # Average over batch

    def _compute_l2_loss(self, scores, pred_top_indices, true_top_indices, true_valid_mask):
        """
        Compute L2 loss: true top classes vs predicted top classes not in true top.
        Only considers valid true classes (ignores padded entries from classes with label 0).

        Args:
            scores: Predicted scores (batch_size, num_classes)
            pred_top_indices: Predicted top class indices (batch_size, top_k)
            true_top_indices: True top class indices (batch_size, top_k)
            true_valid_mask: Mask indicating valid true classes (batch_size, top_k)

        Returns:
            torch.Tensor: L2 loss value
        """
        batch_size, top_k = pred_top_indices.shape

        # Create binary masks for true and predicted top classes
        true_mask = torch.zeros(batch_size, scores.size(1), device=scores.device, dtype=torch.bool)
        pred_mask = torch.zeros(batch_size, scores.size(1), device=scores.device, dtype=torch.bool)

        # Set true mask only for valid true classes
        for batch_idx in range(batch_size):
            valid_true_indices = true_top_indices[batch_idx][true_valid_mask[batch_idx]]
            if len(valid_true_indices) > 0:
                true_mask[batch_idx, valid_true_indices] = True

        # Set predicted mask
        pred_mask.scatter_(1, pred_top_indices, True)

        # Find predicted top classes that are NOT in true top
        pred_not_true_mask = pred_mask & (~true_mask)  # (batch_size, num_classes)

        # Get valid true top indices and their scores
        valid_true_indices = []
        valid_true_scores = []
        max_valid_true = 0

        for batch_idx in range(batch_size):
            batch_valid_indices = true_top_indices[batch_idx][true_valid_mask[batch_idx]]
            valid_true_indices.append(batch_valid_indices)
            max_valid_true = max(max_valid_true, len(batch_valid_indices))

        # Get predicted classes not in true top for each batch
        pred_not_true_indices = []
        max_pred_not_true = 0

        for batch_idx in range(batch_size):
            pnt_indices = torch.where(pred_not_true_mask[batch_idx])[0]
            pred_not_true_indices.append(pnt_indices)
            max_pred_not_true = max(max_pred_not_true, len(pnt_indices))

        if max_pred_not_true == 0 or max_valid_true == 0:
            # No valid pairs for L2 loss
            return torch.tensor(0.0, device=scores.device)

        # Pad and stack valid true indices
        valid_true_padded = torch.zeros(batch_size, max_valid_true,
                                        device=scores.device, dtype=torch.long)
        valid_true_lengths = torch.zeros(batch_size, device=scores.device, dtype=torch.long)

        for batch_idx, indices in enumerate(valid_true_indices):
            if len(indices) > 0:
                valid_true_padded[batch_idx, :len(indices)] = indices
                valid_true_lengths[batch_idx] = len(indices)

        # Pad and stack pred_not_true_indices
        pred_not_true_padded = torch.zeros(batch_size, max_pred_not_true,
                                           device=scores.device, dtype=torch.long)
        pred_not_true_lengths = torch.zeros(batch_size, device=scores.device, dtype=torch.long)

        for batch_idx, indices in enumerate(pred_not_true_indices):
            if len(indices) > 0:
                pred_not_true_padded[batch_idx, :len(indices)] = indices
                pred_not_true_lengths[batch_idx] = len(indices)

        # Get scores for valid true and pred not true classes
        valid_true_scores = scores.gather(1, valid_true_padded)  # (batch_size, max_valid_true)
        pred_not_true_scores = scores.gather(1, pred_not_true_padded)  # (batch_size, max_pred_not_true)

        # Compute pairwise score differences
        true_expanded = valid_true_scores.unsqueeze(2)  # (batch_size, max_valid_true, 1)
        pred_not_true_expanded = pred_not_true_scores.unsqueeze(1)  # (batch_size, 1, max_pred_not_true)

        # Score differences: S(true) - S(pred_not_true)
        score_diffs = true_expanded - pred_not_true_expanded  # (batch_size, max_valid_true, max_pred_not_true)

        # Compute pairwise losses
        pairwise_losses = torch.maximum(torch.zeros_like(score_diffs), -score_diffs + self.delta)

        # Create mask for valid pairs
        valid_true_mask_expanded = (torch.arange(max_valid_true, device=scores.device).unsqueeze(0).unsqueeze(2) <
                                    valid_true_lengths.unsqueeze(1).unsqueeze(2))
        pred_not_true_mask_expanded = (torch.arange(max_pred_not_true, device=scores.device).unsqueeze(0).unsqueeze(0) <
                                       pred_not_true_lengths.unsqueeze(1).unsqueeze(2))

        valid_pairs_mask = valid_true_mask_expanded.float() * pred_not_true_mask_expanded.float()

        # Apply mask and sum
        masked_losses = pairwise_losses * valid_pairs_mask
        l2_loss = masked_losses.sum(dim=(1, 2))  # Sum over all pairs for each batch

        # Count valid pairs for proper averaging
        valid_pair_counts = valid_pairs_mask.sum(dim=(1, 2))  # (batch_size,)
        valid_pair_counts = torch.clamp(valid_pair_counts, min=1)  # Avoid division by zero

        # Average by number of valid pairs per batch
        l2_loss = l2_loss / valid_pair_counts

        return l2_loss.mean()  # Average over batch