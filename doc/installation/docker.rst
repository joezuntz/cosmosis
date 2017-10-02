Docker Installation
-------------------

Docker is a virtual machine-like system that lets us package up all the cosmosis dependencies in a known system, called a "Container".  Working in this container is slightly different to what you may be used to, but is simple enough.

The main advantage here for docker is that it gives a very specific installation environment so any problems you have we can instantly reproduce.  The main disadvantages are disk space - you have to download an *image* of the OS you are using - and the fact that you are working on a different file and operating system than you are used to inside docker.

The docker documentation is terrible for typical scientist users - it mostly assumes you want to run web applications.

You need admin rights to install docker, but not to use it, so if you have a system administrator it is a one-off task for them.


#. Install and start docker for your system

    * `Installation on MacOS <https://docs.docker.com/docker-for-mac/install//>`_ (get the stable version)
    * `Installation on Windows <https://docs.docker.com/docker-for-windows/install/>`_ (get the stable version)
    * On Linux, docker is available in most package managers like apt and yum.  You can also try the more `opaque instructions on the docker site <https://docs.docker.com/engine/installation/#server>`_.

#. :code:`git clone https://bitbucket.org/joezuntz/cosmosis-docker` This will download the cosmosis-docker installer
#. :code:`cd cosmosis-docker`
#. :code:`./get-cosmosis-and-vm ./cosmosis`  This download cosmosis. Wait a little while for the download to complete.
#. :code:`./start-cosmosis-vm ./cosmosis`  #This starts you inside docker.  Read what it says on the screen.
#. First time only: :code:`update-cosmosis --develop` to get the development version which has some fixes in.
#. First time only, or when you change any C/C++/F90 code: :code:`make`
