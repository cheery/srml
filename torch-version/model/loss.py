import torch

def loss_function(model, graph, noise):
    """Returns a loss function for SEDD with HRM.

    The returned function computes the score entropy loss over a batch,
    threading z (HRM state) through as auxiliary output.

    Args:
        model:  nn.Module with signature forward(z, x, sigma) -> (z, log_score)
        graph:  Graph instance
        noise:  Noise instance
    Returns:
        sedd_hrm_loss(z, batch, t=None, perturbed_batch=None) -> (loss, z)
    """
    def sedd_hrm_loss(z, batch, t=None, perturbed_batch=None, memories=None):
        sampling_eps = 1e-3
        device = batch.device

        if t is None:
            t = ((1 - sampling_eps)
                 * torch.rand(batch.shape[0], device=device)
                 + sampling_eps)

        sigma, dsigma = noise(t)

        if perturbed_batch is None:
            perturbed_batch = graph.sample_transition(batch, sigma[:, None])

        z, log_score, aux_loss = model(z, perturbed_batch, sigma, memories=memories)
        loss = graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)
        loss = (dsigma[:, None] * loss).sum(dim=-1)

        return loss.mean() + aux_loss, tuple(zi.detach() for zi in z)

    return sedd_hrm_loss
