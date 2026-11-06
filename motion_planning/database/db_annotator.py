from motion_planning.database import Database
from motion_planning.experiments.utils.mp_sampler import MPSampler
from motion_planning.search import IncrementalPRM
from motion_planning.utils import interpolate_path, smooth_path


def populate_db(
    db: Database,
    mp_sampler: MPSampler,
    num_envs: int,
    num_tasks_per_env: int,
    smooth_paths: bool = False,
    interpolate_paths_delta: float = 0.0,
    prm_num_samples: int = 1000,
    prm_num_neighbors: int = 5,
    retry_empty_paths: bool = False,
):
    for i in range(num_envs):
        env = mp_sampler.sample_env()
        prm = IncrementalPRM(
            env, num_samples=prm_num_samples, num_neighbors=prm_num_neighbors
        )
        prm.create_graph()

        j = 0
        while j < num_tasks_per_env:
            print(f"Env {i + 1}, Task {j + 1}")
            start, target = mp_sampler.sample_task(env)

            path = prm.search(start, target)

            if path is None:
                continue

            if smooth_paths:
                path = smooth_path(env, path)

            if interpolate_paths_delta > 0.0:
                path = interpolate_path(path, env, interpolate_paths_delta)

            if len(path) > 0:
                db.add_path(path)

            if retry_empty_paths and len(path) == 0:
                print(f"Retrying Task: {j + 1}")
                continue

            # Increment Loop Counter
            j += 1
        print(f"DB Size: {len(db)}")


if __name__ == "__main__":
    from motion_planning.obstacle_sets import BiasedPassage
    from motion_planning.space import PointRobot

    path = "saves/database_pr_bp3_large.pickle"

    db = Database()
    mp_sampler = MPSampler(PointRobot(), BiasedPassage, {"num_walls": 3, "bias": 0.5})

    populate_db(
        db,
        mp_sampler,
        num_envs=100,
        num_tasks_per_env=50,
        smooth_paths=True,
        interpolate_paths_delta=0.5,
        retry_empty_paths=True
    )
    db.save_to_path(path)
