Bootstrap Installation
----------------------

Platforms
========================================

We have an installation script which installs cosmosis and all its dependencies that works on 64 bit **Mac OSX Mavericks** (10.9) and **Scientific Linux 6** or its equivalents (**CentOS, Redhat Enterprise**).

Everything this script installs is put in a single directory which can be cleanly deleted and will not interfere with your other installed programs.

Before starting on all platforms
================================

You need :code:`git` and :code:`curl` to run the bootstrap installer.  You can check with these commands::

    curl --help
    git --help

If either gives you a message like "Command not found" then you can install them from `the git website <http://git-scm.com/>`_ or `the curl website <http://curl.haxx.se>`_.

You also need to be running the bash shell.  Check with::

    echo ${SHELL}

If it does not say "bash" then you can either change your shell to bash permanently (search for instructions for your operating system) or manually run :code:`bash` before each time you use CosmoSIS.


Before starting on MacOS
=========================

You need the XCode developer tools on a Mac.  First check if you have them by running::

    xcode-select --install

If the message includes the text :code:`command line tools are already installed` then you can skip to the next section.

Otherwise, first check if you have XCode by running:: 

    which -a gcc

if it includes :code:`/usr/bin/gcc` then you have it already. Otherwise, install XCode from `the apple website <https://developer.apple.com/xcode/>`_ or from the the Mac App Store (it is free).

Next, install the XCode command line tools.  This requires :code:`sudo` powers::

    sudo xcode-select --install


Before starting on Red Hat 6.x, Scientific Linux 6.x, and other derivatives
=============================================================================

On RHEL 6.x derivatives, you need to have the following set of RPM packages installed::

    redhat-lsb-core
    libpng-devel
    freetype-devel
    lapack-devel
    git


You can check this by running::

    yum list redhat-lsb-core libpng-devel freetype-devel lapack-devel git

If any of them appear underneat the "Available packages" section of the output, then you will need to have someone with superuser (admin) privileges install the packages listed using this command::

    su - -c "yum install redhat-lsb-core libpng-devel freetype-devel lapack-devel git"`


Running the bootstrap
==========================

Run these lines to get and run the script::

    curl -L --remote-name https://bitbucket.org/mpaterno/cosmosis-bootstrap/raw/master/cosmosis-bootstrap-linux
    chmod u+x cosmosis-bootstrap
    ./cosmosis-bootstrap cosmosis
    cd cosmosis


Setting up the environment
==========================


Each time you use CosmoSIS you need to first do this::

    source config/setup-cosmosis


Compiling
=========

Build the code like this::

    make

You need to do this again whenever you modify C, C++, or Fortran code.  If you just change python code you don't need to do it again.


Note
====

A note for people who understand git: this procedure leaves you in a "detached head" state at the latest version.  You can get to the bleeding edge with: git checkout master