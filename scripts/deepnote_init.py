###############################################################################
# CS345 DeepNote Setup. DO NOT EDIT.
# Version 0
###############################################################################


import os
import requests


USER_NAME = 'mogu-lab'
REPO_NAME = 'cs345'


def download_directory_from_github(user_name, repo_name, path, skip_downloaded=True):
    '''
    Downloads all files in a directory from a github repository.
    '''
    url = f'https://api.github.com/repos/{user_name}/{repo_name}/contents/{path}'
    directory_info = requests.get(url)
    
    for file_info in directory_info.json():
        print(f'\tLoading {file_info["path"]}...')        
        
        if os.path.exists(file_info['path']) and skip_downloaded:
            print('\tAlready downloaded.')
            continue

        r = requests.get(file_info['download_url'])
        open(file_info['path'] , 'wb').write(r.content)


def download_files_from_github(user_name, repo_name, paths, skip_downloaded=True):
    '''
    Downloads a list of files from a github repository.
    '''
    
    for path in paths:
        print(f'\tLoading {path}...')        
        
        if os.path.exists(path) and skip_downloaded:
            print('\tAlready downloaded.')
            continue

        url = f'https://api.github.com/repos/{user_name}/{repo_name}/contents/{path}'
        file_info = requests.get(url).json()
            
        r = requests.get(file_info['download_url'])
        open(file_info['path'] , 'wb').write(r.content)


def main():
    # Load data
    print('Loading Data:')
    os.makedirs('data', exist_ok=True)
    download_directory_from_github(
        USER_NAME,
        REPO_NAME,
        'data',
        skip_downloaded=False,
    )
    
    # Load helper Python files
    print('Loading helper Python files:')
    download_files_from_github(
        USER_NAME,
        REPO_NAME,
        ['probabilistic_foundations_of_ml.py', 'utils.py'],
        skip_downloaded=False,
    )

    print('Done.')


if __name__ == '__main__':
    main()

