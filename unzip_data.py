import zipfile
import os

zip_file_path = 'archive.zip'
extract_dir = '.'
archive_dir = os.path.join(extract_dir, 'archive')
csv_file_path = os.path.join(archive_dir, 'creditcard.csv')

if os.path.exists(archive_dir) and os.path.exists(csv_file_path):
    print('Data already extracted. Skipping.')
else:
    if not os.path.exists(zip_file_path):
        raise FileNotFoundError(f"'{zip_file_path}' not found. Please make sure the zip file is in the root directory.")
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f'Successfully extracted {zip_file_path} to {extract_dir}')
    
    extracted_folder_name = 'creditcard' # The name of the folder inside the zip
    if os.path.exists(extracted_folder_name):
        os.rename(extracted_folder_name, 'archive')
        print(f'Renamed {extracted_folder_name} to archive')
    else:
        print(f'Folder {extracted_folder_name} not found, skipping rename.')

print('Data extraction and setup complete.')
