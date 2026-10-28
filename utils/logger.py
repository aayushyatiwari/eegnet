import wandb


def init_wandb(cfg, fold, dry_run):
    w = cfg["wandb"]
    return wandb.init(
        project=w["project"],
        entity=w["entity"],
        mode="disabled" if dry_run else w["mode"],
        name=f"{cfg['name']}-fold{fold}",
        group=cfg["name"],
        tags=w["tags"],
        config={**cfg, "fold": fold},
    )
