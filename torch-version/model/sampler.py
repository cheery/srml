import torch
from .catsample import sample_categorical


class EulerPredictor:
    def __init__(self, graph, noise):
        self.graph = graph
        self.noise = noise

    def update_fn(self, score_fn, arg, x, t, step_size):
        sigma, dsigma = self.noise(t)
        arg, log_score = score_fn(arg, x, sigma)
        score = log_score.exp()
        rev_rate = step_size * dsigma[..., None, None] * self.graph.reverse_rate(x, score)
        x = self.graph.sample_rate(x, rev_rate)
        return arg, x


class Denoiser:
    def __init__(self, graph, noise):
        self.graph = graph
        self.noise = noise

    def update_fn(self, score_fn, arg, x, t):
        sigma = self.noise(t)[0]
        arg, log_score = score_fn(arg, x, sigma)
        score = log_score.exp()

        stag_score = self.graph.staggered_score(score, sigma)
        probs = stag_score * self.graph.transp_transition(x, sigma)
        if self.graph.absorb:
            probs = probs[..., :-1]

        return arg, sample_categorical(probs)


class Sampler:
    def sample(
        self,
        score_fn,
        arg,
        graph,
        noise,
        tokenizer,
        batch_size=1,
        batch_len=32,
        steps=1024,
        eps=1e-5,
        denoise=True,
        projector=lambda x: x,
        show_intermediate=False,
        device=None,
    ):
        """Run the full reverse diffusion sampling loop.

        Args:
            score_fn:   callable(arg, x, sigma) -> (arg, log_score)
            arg:        initial model state (e.g. z tuple)
            graph:      Graph instance
            noise:      Noise instance
            tokenizer:  callable(x: int tensor) -> str
            batch_size: number of sequences to generate
            batch_len:  sequence length
            steps:      number of Euler steps
            eps:        small t offset
            denoise:    whether to apply final denoising step
            projector:  optional fn to constrain x between steps
            device:     torch device
        Returns:
            (arg, list[str])
        """
        predictor = EulerPredictor(graph, noise)
        denoiser  = Denoiser(graph, noise)

        x = graph.sample_limit(batch_size, batch_len, device=device)

        timesteps = torch.linspace(1.0, eps, steps + 1, device=device)
        dt = (1 - eps) / steps

        try:
            for i in range(steps):
                t = timesteps[i].expand(batch_size)
                x = projector(x)
                arg, x = predictor.update_fn(score_fn, arg, x, t, dt)

                if show_intermediate:
                    print(f"{i} @ {timesteps[i].item():.5f}:")
                    print(repr([tokenizer(xi) for xi in x]))

        except KeyboardInterrupt:
            pass

        if denoise:
            x = projector(x)
            t = timesteps[-1].expand(batch_size)
            arg, x = denoiser.update_fn(score_fn, arg, x, t)

        if show_intermediate:
            print("Denoised:")
            print(repr([tokenizer(xi) for xi in x]))

        return arg, [tokenizer(xi) for xi in x]
