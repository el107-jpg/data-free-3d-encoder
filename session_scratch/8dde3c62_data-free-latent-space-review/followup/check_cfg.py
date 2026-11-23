"""Check that the item-4 and item-6 overrides parse and give the intended hard limits."""
from arm_planner.data_free.config import DataFreeConfig
from arm_planner.data_free.train import apply_overrides

base = DataFreeConfig()
print("default: inner", base.base_radius - base.correction_bound, "outer", base.base_radius + base.correction_bound,
      "trace target", base.cov_trace_target, "surface edge radius", base.r_s(), "origin margin", base.origin_margin)
cfg = apply_overrides(DataFreeConfig(), ["base_radius=0.45", "correction_bound=0.40",
                                         f"cov_trace_target={base.cov_trace_target}", f"surface_edge_radius={base.r_s()}"])
print("item 4: inner", round(cfg.base_radius - cfg.correction_bound, 4), "outer", round(cfg.base_radius + cfg.correction_bound, 4),
      "trace target", cfg.cov_trace_target, "surface edge radius", cfg.r_s(), "origin margin", cfg.origin_margin,
      "delete_size", cfg.delete_size, "delete_gap", cfg.delete_gap)
cfg6 = apply_overrides(DataFreeConfig(), ["latent_dim=64"])
print("item 6: latent_dim", cfg6.latent_dim)
