# Code TODOs, HACKs, and BUGs

This catalog lists every `TODO`, `HACK`, or `BUG` marker found in the code and notebook files under `motion_planning/`. Entries quote the source comment; repeated markers on one line are listed together.

| # | Location | Marker(s) | Source comment |
|---:|---|---|---|
| 1 | [pure_pursuit.py:38](../motion_planning/controller/pure_pursuit.py#L38) | TODO | `Get the point that this edge intersects with the circle (But for now we will just do the second point)` |
| 2 | [collision_check_ablation.py:11](../motion_planning/experiments/collision_check_ablation.py#L11) | TODO | `Spaces should be able to batch sample points` |
| 3 | [occupancy_map.py:51](../motion_planning/mapping/occupancy_map.py#L51) | TODO | `CHANGE EMPTY TO 0.1?` |
| 4 | [slam.py:44](../motion_planning/mapping/slam.py#L44) | TODO | `Temporary Solution to Plotting Issue` |
| 5 | [dual_cup.py:17](../motion_planning/obstacle_sets/dual_cup.py#L17) | TODO | `Complete the Boundary (Not that hard)` |
| 6 | [informed_rrt_star.py:24](../motion_planning/search/informed_rrt_star.py#L24) | TODO | `Potentially Optimize This Loop` |
| 7 | [informed_rrt_star.py:37](../motion_planning/search/informed_rrt_star.py#L37) | TODO | `Potentially Optimize This Loop` |
| 8 | [pdg.py:53](../motion_planning/search/pdg.py#L53) | TODO | `Need to validate the edge from path to goal` |
| 9 | [pdg.py:167](../motion_planning/search/pdg.py#L167) | TODO | `Figure out logic for updating path states and all` |
| 10 | [pdg.py:214](../motion_planning/search/pdg.py#L214) | HACK | `This might be a hack (may need to figure out why this is happening)` |
| 11 | [pdg.py:229](../motion_planning/search/pdg.py#L229) | TODO / HACK / BUG | `HARDCODED!!!` |
| 12 | [pdg.py:279](../motion_planning/search/pdg.py#L279) | TODO | `Use env dims for the "2" arg` |
| 13 | [rrt.py:111](../motion_planning/search/rrt.py#L111) | TODO / HACK | `Clean This up with constructor (Semi-HACK)` |
| 14 | [rrt.py:114](../motion_planning/search/rrt.py#L114) | TODO / HACK | `Clean This up with constructor (Semi-HACK)` |
| 15 | [lidar.py:91](../motion_planning/sensors/lidar.py#L91) | TODO | `Change the return result to only angle, dist` |
| 16 | [circle_approximation_3d.py:104](../motion_planning/space/approx/circle_approximation_3d.py#L104) | TODO | `Handle Overapproximations` |
| 17 | [circle_approximation_3d.py:130](../motion_planning/space/approx/circle_approximation_3d.py#L130) | TODO | `Handle Overapproximations` |
| 18 | [circle_approximation_3d.py:319](../motion_planning/space/approx/circle_approximation_3d.py#L319) | TODO | `Needs to be implemented properly with representations of 3d obsts` |
| 19 | [circle_approximation_torch.py:151](../motion_planning/space/approx/circle_approximation_torch.py#L151) | TODO | *(no description)* |
| 20 | [circle_approximation_torch.py:159](../motion_planning/space/approx/circle_approximation_torch.py#L159) | TODO | *(no description)* |
| 21 | [circle_approximation_torch.py:165](../motion_planning/space/approx/circle_approximation_torch.py#L165) | TODO | *(no description)* |
| 22 | [circle_approximation_torch.py:181](../motion_planning/space/approx/circle_approximation_torch.py#L181) | TODO | *(no description)* |
| 23 | [disc_robot.py:26](../motion_planning/space/holonomic/disc_robot.py#L26) | TODO | *(no description)* |
| 24 | [disc_robot.py:55](../motion_planning/space/holonomic/disc_robot.py#L55) | TODO | *(no description)* |
| 25 | [disc_robot.py:69](../motion_planning/space/holonomic/disc_robot.py#L69) | TODO | *(no description)* |
| 26 | [fixed_arm.py:106](../motion_planning/space/holonomic/fixed_arm.py#L106) | HACK | `Hack to treat angles properly` |
| 27 | [planar_mobile_arm.py:69](../motion_planning/space/holonomic/planar_mobile_arm.py#L69) | TODO | `Convert to return only an np.ndarray` |
| 28 | [planar_mobile_arm.py:135](../motion_planning/space/holonomic/planar_mobile_arm.py#L135) | HACK | `Hack to treat angles properly` |
| 29 | [planar_mobile_arm.py:291](../motion_planning/space/holonomic/planar_mobile_arm.py#L291) | HACK | `Hack to treat angles properly` |
| 30 | [polygonal_robot.py:133](../motion_planning/space/holonomic/polygonal_robot.py#L133) | TODO | `Standardize the returnable for this function` |
| 31 | [dubins_car.py:178](../motion_planning/space/nonholonomic/dubins_car.py#L178) | HACK | `Hack to treat the upward direction` |
| 32 | [mpc.py:138](../motion_planning/trajectory_optimization/mpc.py#L138) | TODO | `Deal with this case` |
| 33 | [mpc.py:246](../motion_planning/trajectory_optimization/mpc.py#L246) | TODO | `Uncomment` |
| 34 | [utils.py:134](../motion_planning/utils/utils.py#L134) | TODO | `Probably need to reshape # (B, d)` |
| 35 | [viz_cspace_2d.py:110](../motion_planning/visualizations/viz_cspace_2d.py#L110) | TODO / HACK | `DO NOT CHANGE ARM LENGTHS LIKE THIS` |
| 36 | [viz_cspace_2d.py:203](../motion_planning/visualizations/viz_cspace_2d.py#L203) | TODO | `Fix Controller Integration` |
| 37 | [test_utils_real.py:32](../motion_planning/tests/unit_tests/test_utils_real.py#L32) | TODO | `Ensure angular wrap around is tested here` |
