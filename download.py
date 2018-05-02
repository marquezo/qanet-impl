#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def download_glove():
    
    import zipfile

    !wget http://nlp.stanford.edu/data/glove.6B.zip
    
    zip_ref = zipfile.ZipFile('glove.6B.zip.1', 'r')
    zip_ref.extractall()
    zip_ref.close()
    
# =============================================================================
#     import zipfile
#     from squad_preprocess import *
#     
#     if __name__ == '__main__':
#         glove_base_url = "http://nlp.stanford.edu/data/"
#         glove_filename = "glove.6B.zip"
#         prefix = os.path.join("download", "dwr")
#     
#         print("Storing datasets in {}".format(prefix))
#     
#         if not os.path.exists(prefix):
#             os.makedirs(prefix)
#     
#         glove_zip = maybe_download(glove_base_url, glove_filename, prefix, 862182613L)
#         glove_zip_ref = zipfile.ZipFile(os.path.join(prefix, glove_filename), 'r')
#     
#         glove_zip_ref.extractall(prefix)
#         glove_zip_ref.close()
# =============================================================================


def download_squad():
    
    !wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
    !wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json



if __name__ == "__main__":
  
    download_glove()
    download_squad()
