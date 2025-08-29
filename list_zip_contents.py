import zipfile

zip_file_path = 'archive.zip'

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.printdir()
