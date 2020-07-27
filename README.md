# Faceable Client Application

This repository contains the source code for the client desktop application for the `Faceable` project as the submission entry for `Microsoft Garage Hackathon 2020` under the category `Hack for Security of Remote Work` to avoid shoulder surfing.

[![Architecture](https://i.ibb.co/swRCrcN/Face-ID-Garage-Schema-v4.jpg)](https://nodesource.com/products/nsolid)

## Screenshots

## Features
The source code handles the following features:
* Checking current public IP address of the machine
* Checking Hostname and Mac Address
* Collecting dataset for training the on-device Face Recognition model
* Orchestrating with backend API layer to control the rules engine
* Call Azure Face Recognition API to verify the user
* Continous recognition on-device with minimal compute resources

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.
### Prerequisites

What things you need to install the software and how to install them

* [Python 3.6](https://www.python.org/downloads/release/python-360/) - Python is an interpreted, high-level, general-purpose programming language
* [Kivy 1.11.1](https://kivy.org/doc/stable/installation/installation-windows.html) - Open source Python library for rapid development of applications
that make use of innovative user interfaces, such as multi-touch apps.
* [Inno Setup](https://jrsoftware.org/isdl.php) - Inno Setup is a free installer for Windows programs 

### Running locally

##### Install the necessary packages
And installing dependencies
```
pip install -r requirements.txt
```
##### Running locally
For local development
```
python faceable.py
```

#### Creating a Release
##### Creating a executable file (.exe) from .py 

To create an executable file from Python follow the steps below

In the `faceable.spec` file change line no `34,35,36 paths` to the respective paths where the `gstreamer`, `glew` and `sdl2` are installed

Then run the following command

```
python -m PyInstaller faceable.spec
```

When the command successfully finishes running, a `dist` folder will be created in the root directory. This folder contains the distribution packages and the `faceable.exe` file

#### Creating a Windows installer

To create a Windows installer, first follow the above steps then follow the steps below 

1. Go to the folder that contains the .iss file

2. Open the .iss file and enter your `Ouput dir` and give the appropriate `Source`

3. The `Output dir` path is where the installer file will get stored and the `Source` indicates the folder and the .exe to include when creating the Windows installer

4. Then, Click on Build -> Start Compile

5. After successful compilation, the `.exe` file will be generated in the `Output dir` specified

#### Running the Release

Once the installer file is generated following the steps above, run the installer file. On successful installation, `run the installed program as an administrator.`

## Security Aspects
The following steps are handled to ensure security is maintained for the user
* All data is wiped after training
* No processes or user activity is tracked at all
* Backend APIs are called securely and the endpoints are hidden after the release gets created
* The Camera service is disabled when the system is in locked mode.

## Development Team
* Pourab Karchaudhuri
* Rowland Arul Jonathan
* Pournima Mishra
* Anlin Jose
* Mayank Sahu
* Shubham Gupta
