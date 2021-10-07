#!/bin/bash

# remove old generated files
rm -rf build dist
mkdir -p build
mkdir -p dist

# make the .app
#py2applet --make-setup app.py
poetry run python setup.py py2app

# move data files into app
cp -r html dist/app.app/Contents/Resources/
cp cert.pem dist/app.app/Contents/Resources/
