from motion_planning.database import Database
from motion_planning.utils import smooth_path, interpolate_path
from motion_planning.experiments.utils.mp_sampler import MPSampler
from motion_planning.search import IncrementalPRM

def populate_db(db: Database, mp_sampler: MPSampler, num_envs: int, num_tasks_per_env: int, smooth_paths: bool = False, interpolate_paths_delta: float = 0.0):
    for i in range(num_envs):
        env = mp_sampler.sample_env()
        prm = IncrementalPRM(env, num_samples=1000, num_neighbors=5)
        prm.create_graph()

        for j in range(num_tasks_per_env):
            print(f"Env {i+1}, Task {j+1}")
            start, target = mp_sampler.sample_task(env)

            path = prm.search(start, target)

            if path is None:
                continue

            if smooth_paths:
                path = smooth_path(env, path)
            
            if interpolate_paths_delta > 0.0:
                path = interpolate_path(path, env, interpolate_paths_delta)

            db.add_path(path)
        print(f"DB Size: {len(db)}")