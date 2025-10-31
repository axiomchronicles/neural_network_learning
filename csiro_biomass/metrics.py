import torch

def r2_score_fn(y_true, y_pred):
    # Computes the coefficient of determination (R²) for each output dimension.

    # Mathematical Formula:
    #     R² = 1 - (SS_res / SS_tot)

    # where:
    #     SS_res = Σᵢ (yᵢ - ŷᵢ)²          → Residual Sum of Squares
    #     SS_tot = Σᵢ (yᵢ - ȳ)²           → Total Sum of Squares
    #     ȳ = (1/n) * Σᵢ yᵢ               → Mean of actual values

    # Interpretation:
    #     - R² = 1   → Perfect fit (model explains all variance)
    #     - R² = 0   → Model predicts no better than the mean
    #     - R² < 0   → Model performs worse than mean predictor

    ss_res = torch.sum((y_true - y_pred) ** 2, dim=0)
    ss_tot = torch.sum((y_true - torch.mean(y_true, dim=0)) ** 2, dim=0)
    r2 = 1 - ss_res / ss_tot
    return r2


def weighted_r2(y_true, y_pred):
    # Computes a weighted R² score across multiple regression targets.

    # Mathematical Formula:
    #     Weighted R² = Σⱼ (wⱼ * R²ⱼ)
    
    # where:
    #     wⱼ → weight for the j-th target
    #     R²ⱼ → coefficient of determination for the j-th target

    # This emphasizes some targets more than others, depending on their weights.

    weights = torch.tensor([0.1, 0.1, 0.1, 0.5, 0.2], device=y_true.device)
    r2 = r2_score_fn(y_true, y_pred)
    final = torch.sum(weights * r2)
    return r2, final
