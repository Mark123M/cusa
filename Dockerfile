FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        xfce4 \
        xfce4-goodies \
        x11vnc \
        xvfb \
        xclip \
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
RUN mkdir -p /tmp/.X11-unix && chmod 1777 /tmp/.X11-unix
USER myuser
WORKDIR /home/myuser

RUN x11vnc -storepasswd secret /home/myuser/.vncpass

EXPOSE 5900
CMD ["/bin/sh", "-c", "\
    set -eu; \
    rm -f /tmp/.X99-lock /tmp/.X11-unix/X99; \
    Xvfb :99 -screen 0 1920x1080x24 >/tmp/xvfb.log 2>&1 & xvfb_pid=$!; \
    sleep 1; \
    kill -0 $xvfb_pid; \
    x11vnc -display :99 -forever -shared -rfbauth /home/myuser/.vncpass -listen 0.0.0.0 -rfbport 5900 -o /tmp/x11vnc.log >/tmp/x11vnc.stdout.log 2>&1 & vnc_pid=$!; \
    export DISPLAY=:99 && \
    dbus-launch --exit-with-session startxfce4 >/tmp/xfce.log 2>&1 & xfce_pid=$!; \
    sleep 2; \
    kill -0 $vnc_pid; \
    kill -0 $xfce_pid; \
    echo 'Container running! VNC on port 5900'; \
    tail -F /tmp/xvfb.log /tmp/x11vnc.log /tmp/x11vnc.stdout.log /tmp/xfce.log \
"]
