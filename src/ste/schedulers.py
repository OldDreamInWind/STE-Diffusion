from diffusers import DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler


def build_noise_scheduler(
    scheduler_type: str = "ddpm",
    num_train_timesteps: int = 1000,
    beta_schedule: str = "linear",
):
    """Build a training noise scheduler. Only DDPM is supported for training."""
    if scheduler_type == "ddpm":
        return DDPMScheduler(num_train_timesteps=num_train_timesteps, beta_schedule=beta_schedule)
    raise ValueError(f"Unsupported training scheduler: {scheduler_type!r}. Only 'ddpm' is supported.")


def build_inference_scheduler(
    scheduler_type: str = "ddpm",
    num_train_timesteps: int = 1000,
    beta_schedule: str = "linear",
):
    """Build an inference scheduler.

    Supported: ddpm, ddim, dpmsolver (alias: dpms).
    """
    t = scheduler_type.lower()
    if t == "ddpm":
        return DDPMScheduler(num_train_timesteps=num_train_timesteps, beta_schedule=beta_schedule)
    if t == "ddim":
        return DDIMScheduler(num_train_timesteps=num_train_timesteps, beta_schedule=beta_schedule)
    if t in {"dpmsolver", "dpms"}:
        return DPMSolverMultistepScheduler(num_train_timesteps=num_train_timesteps)
    raise ValueError(f"Unknown inference scheduler: {scheduler_type!r}. Choose: ddpm, ddim, dpmsolver.")
