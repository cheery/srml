import jax
import jax.numpy as jnp

def loss_function(model, graph, noise):
    def sedd_hrm_loss(params, key, z, batch, t=None, perturbed_batch=None):
        sampling_eps = 1e-3

        if t is None:
            key, t_key = jax.random.split(key)
            t = (1 - sampling_eps) * jax.random.uniform(t_key, (batch.shape[0],)) + sampling_eps

        sigma, dsigma = noise(t)

        if perturbed_batch is None:
            key, trans_key = jax.random.split(key)
            perturbed_batch = graph.sample_transition(trans_key, batch, sigma[:, None])

        z, log_score = model.apply(params, key, z, perturbed_batch, sigma)
        loss = graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)
        loss = (dsigma[:, None] * loss).sum(axis=-1)

        return loss.mean().astype(jnp.float32), jax.lax.stop_gradient(z)
    return sedd_hrm_loss

# a small code to check what is going on.
        #from .catsample import sample_categorical
        #print("TIME:", t[0])
        #print("BATCH:", repr(as_text(batch[0])))
        #print("PERT:", repr(as_text(perturbed_batch[0])))
        #batch_b = perturbed_batch[0,None]
        #score_b = np.exp(log_score[0,None])
        #sigma_b = sigma[0,None]
        #stag_score_b = graph.staggered_score(score_b, sigma_b)
        #probs_b = stag_score_b * graph.transp_transition(batch_b, sigma_b)
        #probs_b = probs_b[..., :-1]
        #solu_b = sample_categorical(key, probs_b)
        #print("SOLU:", repr(as_text(solu_b[0])))

