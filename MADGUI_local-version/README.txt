This is a local version of MADGUI, Multi-Application Design Graphical user Interface, you can use it if the online version is too slow or if you don't want to upload your data on the online server of streamlit.

If you have question or bug report you can use the GitHub or the online MADGUI contact page to contact us.

If you want to use the local version, download the zip by clicking "Code" -> "Download ZIP"

To launch MADGUI you have to open your terminal then go to the repertory where you save the MADGUI.py file.

For exemple, if you save the folder on your Desktop, the line you have to write is :

cd Desktop/MADGUI_local-version   on MAC
cd Desktop\MADGUI_local-version   on Windows

Requieres Python >= 3.8 and <3.11 (tested succesfully on python 3.8.10 and 3.9.6) (tested unsuccesfully on python 3.7.9, 3.11.7 and 3.12.1, limitted by GpyOpt version)

Then for the first use you have to install the requirements packages neccessary for the use of MADGUI, it can take some times:

pip install -r requirements.txt

Then you can use MADGUI every time you want it by writing :

streamlit run MADGUI.py

