def get_temp_path(fileCacheManager, pid, rnd_id, filename):
    import os
    temp_dir = os.path.join(fileCacheManager.cache_dir, f"tmp.pid_{pid}_{rnd_id}")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, filename)
    os.removedirs(temp_dir)
    return temp_path


def remove_temp_dir(fileCacheManager, pid, rnd_id):
    import os
    temp_dir = os.path.join(fileCacheManager.cache_dir, f"tmp.pid_{pid}_{rnd_id}")
    os.removedirs(temp_dir)