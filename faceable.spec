# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['faceable.py'],
             pathex=['your_root_directory_path'],
             binaries=[],
             datas=[('caffe/deploy.prototxt.txt','caffe'),('caffe/res10_300x300_ssd_iter_140000.caffemodel','caffe'),('static/black.png','static'),('static/faceable_logo.png','static'),('static/faceable_logo.ico','static'),('dataset/unknown','dataset/unknown'),('openface_nn4.small2.v1.t7','.'),('screens.kv','.'),('static/loading.gif','static'),('Faceable_test_cd.xml','.'),('schedule.bat','.')],
             hiddenimports=['sklearn.utils._cython_blas'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='faceable',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False)
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               Tree('your_path_to_Python\\Python36\\share\\sdl2\\bin'),
               Tree('your_path_to_Python\\Python\\Python36\\share\\glew\\bin'),
               Tree('your_path_to_Python\\Python\\Python36\\share\\gstreamer\\bin'),
               strip=False,
               upx=True,
               upx_exclude=[],
               name='faceable')
