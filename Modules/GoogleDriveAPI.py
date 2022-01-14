#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class gDrive(object):

    def __init__(self):
        from pydrive.auth import GoogleAuth
        from pydrive.drive import GoogleDrive
        
        #Acessando e autenticando o GoogleDrive
        print('\nAutenticando o Google Drive...')
        gauth = GoogleAuth()
        gauth.LocalWebserverAuth() # client_secrets.json need to be in the same directory as the script
        self.drive = GoogleDrive(gauth)    
    

    def upload(self, file_name_to_upload, folder_path, folderID='', fileType = "text/csv"):
        # View all folders and file in your Google Drive
        fileList = self.drive.ListFile({'q': "'" + folderID + "' in parents and trashed=false"}).GetList()
        #procurando o fileID do arquivo salvo para deletá-lo
        for file in fileList:
            if(file['title'] == file_name_to_upload):
                fileID = file['id']
                file2 = self.drive.CreateFile({'id': fileID})
                file2.Trash()  # Move file to trash.
                print('\nArquivo encontrado e enviado para a lixeira:')
                print('Title: %s, ID: %s' % (file['title'], file['id']))
                #file2.UnTrash()  # Move file out of trash.
                #file2.Delete()  # Permanently delete the file.            
    
        #salvando o arquivo novo
        file_to_upload = self.drive.CreateFile({"title": file_name_to_upload,
                                                "mimeType": fileType, 
                                                "parents":[{"kind": "drive#fileLink", "id": folderID}]})
        file_to_upload.SetContentFile(folder_path + '/' + file_name_to_upload)
        file_to_upload.Upload() # Upload the file.
        print('\nNovo arquivo salvo:')
        print('Created file %s with mimeType %s' % (file_to_upload['title'], file_to_upload['mimeType']))
        
        

#Supported files in GoogleAPI
'''
SUPPORTED_FILETYPES = {
  'CSV': 'text/csv',
  'TSV': 'text/tab-separated-values',
  'TAB': 'text/tab-separated-values',
  'DOC': 'application/msword',
  'DOCX': ('application/vnd.openxmlformats-officedocument.'
           'wordprocessingml.document'),
  'ODS': 'application/x-vnd.oasis.opendocument.spreadsheet',
  'ODT': 'application/vnd.oasis.opendocument.text',
  'RTF': 'application/rtf',
  'SXW': 'application/vnd.sun.xml.writer',
  'TXT': 'text/plain',
  'XLS': 'application/vnd.ms-excel',
  'XLSX': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
  'PDF': 'application/pdf',
  'PNG': 'image/png',
  'PPT': 'application/vnd.ms-powerpoint',
  'PPS': 'application/vnd.ms-powerpoint',
  'HTM': 'text/html',
  'HTML': 'text/html',
  'ZIP': 'application/zip',
  'SWF': 'application/x-shockwave-flash'
  }
'''