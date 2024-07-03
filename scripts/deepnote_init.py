import os
import requests

USER_NAME = 'mogu-lab'
REPO_NAME = 'cs349-fall-2024'


def download_directory_from_github(user_name, repo_name, path, skip_downloaded=True):
    url = f'https://api.github.com/repos/{user_name}/{repo_name}/contents/{path}'
    directory_info = requests.get(url)
    
    for file_info in directory_info.json():
        if os.path.exists(file_info['path']) and skip_downloaded:
            continue

        print(f'\tLoading {file_info["path"]}...')        
        r = requests.get(file_info['download_url'])
        open(file_info['path'] , 'wb').write(r.content)


def download_files_from_github(user_name, repo_name, paths, skip_downloaded=True):
    for path in paths:
        if os.path.exists(path) and skip_downloaded:
            continue

        print(f'\tLoading {path}...')        
        
        url = f'https://api.github.com/repos/{user_name}/{repo_name}/contents/{path}'
        file_info = requests.get(url).json()
            
        r = requests.get(file_info['download_url'])
        open(file_info['path'] , 'wb').write(r.content)


# Load data
print('Loading Data:')

os.makedirs('data', exist_ok=True)
download_directory_from_github(USER_NAME, REPO_NAME, 'data')
download_files_from_github(USER_NAME, REPO_NAME, ['cs349.py', 'utils.py'])

print('Done.')
