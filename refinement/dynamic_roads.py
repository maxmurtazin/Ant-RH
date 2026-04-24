"""
dynamic_roads.py

Formula reference for DTES dynamic roads.

R_t^color(u,v) =
    alpha * log(tau_shared(u,v))
  + alpha_c * log(tau_color(u,v))
  - beta * barrier(u,v)
  + gamma * exploration_bonus(v)
  + delta * color_policy_bonus(color,u,v)

Color policies:
    red     - exploit low-energy / high-score points
    blue    - prefer under-visited regions
    green   - prefer boundary regions
    violet  - prefer gaps between selected candidates
"""
