FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        xfce4 \
        xfce4-goodies \
        x11vnc \
        xvfb \
        xdotool \
        wmctrl \
        imagemagick \
        x11-apps \
        dbus-x11 \
        sudo \
        gnupg \
        software-properties-common \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get remove -y light-locker xfce4-screensaver xfce4-power-manager || true

RUN add-apt-repository -y ppa:mozillateam/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends firefox-esr \
    && update-alternatives --set x-www-browser /usr/bin/firefox-esr \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# CUA workload apps: native multi-window applications for cross-app tasks.
# Separate layer so the app list can change without rebuilding the base
# desktop. Matches the base style with --no-install-recommends to stay lean
# (these still inherit the fonts/icons/theme the xfce4 layer pulled in); drop
# the flag for an individual app if it misrenders or lacks codecs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        xterm \
        gnumeric \
        libreoffice-calc \
        libreoffice-writer \
        mousepad \
        gedit \
        evince \
        ristretto \
        xarchiver \
        thunar-archive-plugin \
        sqlitebrowser \
        vlc \
        gimp \
        galculator \
        thunderbird \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -ms /bin/bash myuser     && echo "myuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER myuser
WORKDIR /home/myuser

RUN x11vnc -storepasswd secret /home/myuser/.vncpass

EXPOSE 5900
CMD ["/bin/sh", "-c", "\
    Xvfb :99 -screen 0 1920x1080x24 >/dev/null 2>&1 & \
    x11vnc -display :99 -forever -rfbauth /home/myuser/.vncpass -listen 0.0.0.0 -rfbport 5900 >/dev/null 2>&1 & \
    export DISPLAY=:99 && \
    dbus-launch --exit-with-session startxfce4 >/dev/null 2>&1 & \
    sleep 2 && echo 'Container running!' && \
    tail -f /dev/null \
"]
